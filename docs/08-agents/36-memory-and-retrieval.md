# 36 — chunking、vector search、引用付き retrieval

## この章で作るもの

文書を source offset 付き chunk に分け、deterministic hashing vector へ変換し、cosine similarity で
検索して citation metadata とともに agent tool へ返します。

対象コードは `src/main/scala/learnai/retrieval/Retrieval.scala` です。

## context window と external memory

model weight は学習時知識を圧縮していますが、最新・private・詳細な文書を正確に保持する database では
ありません。全資料を prompt に入れると context/cost/attention が増えます。query に関連する小部分だけ
を取得します。

```text
documents -> chunks -> embeddings -> index
query     -> embedding -> similarity -> top chunks -> model context
```

retrieval は model parameter を変更しません。検索結果も untrusted data であり、tool 権限にはなりません。

## chunk と provenance

各 `TextChunk` は text だけでなく次を保持します。

- stable chunk/document ID
- document title
- source start/end offset

回答時に chunk ID を引用し、後から原文 span を検証できます。text だけを copy すると source mapping を
失い、誤引用や更新追跡が難しくなります。

固定文字数 chunk は単純ですが、文・見出し・code block を分断します。production では structure-aware
chunking、token count 上限、parent/child retrieval を比較します。

## overlap

境界をまたぐ語や文脈を両 chunk に含めるため overlap を使います。chunk size \(C\)、overlap \(O\) の
step は \(C-O\) です。overlap を増やすと recall は上がり得ますが、index size と重複 result が増えます。

## hashing embedding

教材 embedder は lower-case token の stable hash を固定 dimension bucket へ加え、L2 normalize します。

\[
\hat{v}=v/\lVert v\rVert_2
\]

同じ語の overlap を学ぶ bag-of-words baseline で、synonym や文脈的意味は理解しません。hash collision も
あります。signed hashing は collision の正方向 bias を減らします。

semantic embedding model を導入する前に、index/query vector、normalization、ranking、citation、eval の
pipeline を観察できます。

## cosine search

query/chunk vector を normalize 済みなので cosine similarity は内積です。

\[
\cos(q,d)=\hat{q}\cdot\hat{d}
\]

score 降順、同点は chunk ID 昇順で deterministic にします。ANN index は大規模検索を近似高速化します
が、まず exact scan を evaluation reference にします。

## retrieval tool

`SearchTool` は read-only capability で、query を検証後、各 result に score と source metadata を返し
ます。agent は result text を根拠に回答できますが、runtime/UI は次を強制できます。

- cited chunk ID が実在する
- answer claim と citation span を表示する
- retrieval failure 時に知識を捏造せず不足を示す
- retrieved instruction を system policy として扱わない

## retrieval evaluation

end-to-end answer だけでなく段階別に測ります。

- Recall@k: 正解根拠が top-k に入る割合
- MRR: 最初の正解順位の逆数
- citation precision/coverage
- answer faithfulness
- latency と index bytes

chunk size/overlap/embedder/top-k を変え、固定 query-ground-truth set で比較します。

## 演習

1. sentence boundary を優先する chunker を実装してください。
2. hashing dimension を変え、collision と Recall@k を測ってください。
3. BM25 lexical baseline を実装し、hashing cosine と比較してください。
4. retrieved text に prompt injection を入れ、tool capability が増えないことを test してください。
5. result chunk ID だけを許可する citation validator を追加してください。

## 完了条件

- model weight と external retrieval memory を区別できる
- chunk offset/provenance が引用に必要な理由を説明できる
- overlap の recall/storage trade-off を説明できる
- normalized cosine が dot product になることを説明できる
- retrieval quality と answer quality を分けて評価できる
- `RetrievalSuite` が成功する
