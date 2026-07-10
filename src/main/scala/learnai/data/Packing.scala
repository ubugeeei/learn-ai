package learnai.data

import learnai.text.TokenId

/** One fixed-length packed training window with a per-position loss mask.
  *
  * `lossMask(t)` is true exactly when the prediction `inputs(t) ->
  * targets(t)` should contribute loss and gradient. Masked positions keep
  * the window shape fixed — which batching requires — while excluding
  * padding targets and cross-document predictions from training.
  */
final case class PackedExample(
    inputs: Vector[TokenId],
    targets: Vector[TokenId],
    lossMask: Vector[Boolean]
):
  require(inputs.nonEmpty, "a packed example cannot be empty")
  require(
    inputs.size == targets.size && inputs.size == lossMask.size,
    s"input/target/mask lengths differ: ${inputs.size}/${targets.size}/${lossMask.size}"
  )

  val contextLength: Int = inputs.size

  /** Number of positions that train the model. May be zero for degenerate
    * windows (for example a context of one whose only input is a
    * separator); consumers should batch or filter with
    * [[PackingResult.trainableExamples]].
    */
  def unmaskedTargetCount: Int = lossMask.count(identity)

/** The complete result of packing one document collection.
  *
  * The counts exist so tests and experiment manifests can assert the
  * packing arithmetic instead of trusting it: `unmaskedTargetCount` must
  * equal the total document token count, because every document token is
  * predicted exactly once (its predecessor predicts it, or, for a
  * document's final token, it predicts the separator).
  */
final case class PackingResult(
    contextLength: Int,
    examples: Vector[PackedExample],
    packedStreamLength: Int,
    paddingTokenCount: Int,
    unmaskedTargetCount: Int
):
  /** Windows that contain at least one trainable target. */
  def trainableExamples: Vector[PackedExample] =
    examples.filter(_.unmaskedTargetCount > 0)

object SequencePacking:
  /** Packs variable-length documents into fixed windows with loss masks.
    *
    * The documents are concatenated in order, each terminated by
    * `separator`, into one stream. The stream is cut into consecutive
    * windows of `contextLength + 1` tokens advancing by `contextLength`,
    * so every next-token prediction in the stream appears in exactly one
    * window. The tail is completed with `padding`.
    *
    * The mask at window position `t` is false when:
    *   - the target token is padding (the stream has ended), or
    *   - the input token is the separator — that prediction would teach the
    *     model to guess the *next document's* opening from the previous
    *     document's context.
    *
    * Predicting the separator itself stays unmasked: learning where
    * documents end is real signal. Note what packing does *not* do here:
    * attention inside a window still crosses document boundaries, because
    * this course's attention has no block-diagonal document mask yet. That
    * contamination is bounded by the window length and is the standard
    * trade-off of simple packing; making it explicit is the point.
    *
    * Documents must not contain the separator or padding tokens, and the
    * two must differ, so mask semantics stay unambiguous.
    */
  def pack(
      documents: Vector[Vector[TokenId]],
      contextLength: Int,
      separator: TokenId,
      padding: TokenId
  ): PackingResult =
    require(contextLength > 0, s"context length must be positive: $contextLength")
    require(documents.nonEmpty, "packing requires at least one document")
    require(separator != padding, "separator and padding tokens must differ")
    documents.zipWithIndex.foreach { case (document, index) =>
      require(document.nonEmpty, s"document $index is empty")
      require(
        !document.contains(separator),
        s"document $index contains the separator token"
      )
      require(
        !document.contains(padding),
        s"document $index contains the padding token"
      )
    }

    val stream = documents.flatMap(document => document :+ separator)
    val predictionCount = stream.size - 1
    val windowCount = (predictionCount + contextLength - 1) / contextLength
    val paddedLength = windowCount * contextLength + 1
    val padded = stream ++ Vector.fill(paddedLength - stream.size)(padding)

    val examples = Vector.tabulate(windowCount) { window =>
      val offset = window * contextLength
      val inputs = padded.slice(offset, offset + contextLength)
      val targets = padded.slice(offset + 1, offset + contextLength + 1)
      val mask = Vector.tabulate(contextLength) { position =>
        val absolute = offset + position
        val targetIsReal = absolute + 1 < stream.size
        targetIsReal && padded(absolute) != separator
      }
      PackedExample(inputs, targets, mask)
    }

    PackingResult(
      contextLength = contextLength,
      examples = examples,
      packedStreamLength = stream.size,
      paddingTokenCount = paddedLength - stream.size,
      unmaskedTargetCount = examples.iterator.map(_.unmaskedTargetCount).sum
    )
