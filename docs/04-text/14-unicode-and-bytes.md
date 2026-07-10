# 14 — Unicode, UTF-8, and a byte tokenizer

## What you will build

Convert a Scala `String` to UTF-8 byte token IDs and reject malformed byte
sequences instead of silently replacing them. Source:
`src/main/scala/learnai/text/Utf8.scala`.

## Models do not consume strings directly

Neural networks consume numbers. A tokenizer is the boundary between text and
token IDs:

```text
text --encode--> token IDs --model--> token IDs --decode--> text
```

Tokenization affects vocabulary size, sequence length, training efficiency,
multilingual behavior, generated text, and model parameter count.

## What is a character?

Several units are easily confused:

- **grapheme cluster**: what a reader perceives as one symbol;
- **Unicode code point**: a number such as `U+03B1`;
- **UTF-16 code unit**: a 16-bit unit exposed by JVM string APIs;
- **UTF-8 byte**: an 8-bit unit widely used in files and networks.

One visible symbol may contain several code points. The same visible text may
also have different normalized code-point sequences. `String.length` is not a
universal character count.

## Unicode and UTF-8

Unicode assigns code points. UTF-8 encodes a sequence of code points as bytes.
A code point uses one to four UTF-8 bytes.

| Text | Code point | UTF-8 bytes | Byte tokens |
| --- | --- | --- | --- |
| `A` | `U+0041` | `41` | 1 |
| `α` | `U+03B1` | `CE B1` | 2 |
| `🚀` | `U+1F680` | `F0 9F 9A 80` | 4 |

On the JVM, `Byte` is signed. `byte & 0xff` converts it to the unsigned token
range `[0,255]`.

## Why byte tokenization works for every UTF-8 text

Assign token IDs `0..255` directly to byte values:

- fixed, small vocabulary;
- no unknown token for valid UTF-8;
- reversible encoding;
- simple implementation.

The cost is sequence length. Many scripts and emoji require multiple bytes per
visible symbol. Attention cost grows quadratically with sequence length, so the
next chapter merges frequent byte sequences with BPE.

## Special tokens

Structural tokens must not collide with encoded text:

```text
0..255: byte values
256:    beginning of text
257:    end of text
```

Production protocols may also have role, padding, and tool-call tokens. Adding
one changes vocabulary size and therefore embedding/output shapes.

## Strict decoding

Not every byte sequence is valid UTF-8. `0x80` alone is a continuation byte
without a leader. A replacement-character decoder would hide that the model
generated an invalid sequence.

This course configures `CodingErrorAction.REPORT` and returns
`Either[String,String]`. A streaming UI may buffer an incomplete suffix until
the next token, but raw tokens and errors must remain observable.

## Round-trip invariant

For valid text:

\[
\operatorname{decode}(\operatorname{encode}(text))=text
\]

Test ASCII, several scripts, emoji, combining marks, newlines, and the empty
string. This tokenizer performs no normalization, so it preserves original
UTF-8 bytes.

## Opaque token IDs

```scala
opaque type TokenId = Int
```

Runtime representation is `Int`, but the API has a distinct semantic type.
Lengths and offsets are less likely to be passed accidentally, and construction
rejects negative IDs.

## Implementation walkthrough

`Utf8.encodeBytes` delegates character-to-byte conversion to the JDK's UTF-8
implementation, then maps signed JVM bytes into unsigned integers with
`byte & 0xff`. This is why base token IDs occupy exactly `0..255` even though
Java `Byte` ranges from `-128..127`.

`decodeBytes` reverses the conversion only after validating every integer is in
the byte range. A `CharsetDecoder` is configured to report malformed and
unmappable input instead of inserting the replacement character. Silent
replacement would make `decode(encode(text))` look successful while corrupting
invalid token sequences.

Work one example. ASCII `A` becomes one byte `0x41`. The euro sign becomes
three bytes `E2 82 AC`; the rocket emoji becomes four bytes `F0 9F 9A 80`.
Visible-character count, Unicode code-point count, UTF-16 code-unit count, and
UTF-8 byte count are therefore different quantities.

`ByteTokenizer.encode` maps each unsigned byte to `TokenId` and optionally adds
begin/end special IDs outside `0..255`. Decode either skips known specials or
rejects them, depending on the explicit flag. `TokenId` prevents negative IDs
at construction but vocabulary-specific upper bounds remain the model or
tokenizer's responsibility.

## Reading the tests

Round trips include ASCII, accented Latin text, and emoji so one-byte-only code
cannot pass. Explicit byte counts exercise one-, three-, and four-byte cases.
A lone continuation byte is a compact malformed UTF-8 oracle. Special-token
tests cover both skip and strict behavior. Negative ID rejection protects every
downstream embedding lookup.

## Debugging checklist

1. Print bytes as unsigned hexadecimal, not signed decimal.
2. Distinguish UTF-16 `String.length` from Unicode code points and UTF-8 bytes.
3. Configure decoder error actions before decoding untrusted bytes.
4. If round trip fails, compare raw bytes before tokenizer IDs.
5. Keep special-token IDs outside the byte vocabulary and version their meaning.

## Exercises

1. Encode your name and print each byte in hexadecimal.
2. Add round trips for empty text, newline, and NUL.
3. Compare `String.length`, code-point count, and UTF-8 byte count.
4. Decode incomplete prefixes of a multi-byte symbol.
5. Design a decoder result that preserves special tokens structurally.

## Completion criteria

- Distinguish grapheme, code point, UTF-8 byte, and token.
- Explain why a byte tokenizer has no unknown token.
- Explain why some scripts use longer byte sequences than ASCII.
- Explain why strict decoding helps incident analysis.
- Verify multilingual round trips.
- `Utf8Suite` passes.
