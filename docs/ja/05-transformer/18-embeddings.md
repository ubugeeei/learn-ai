# 18 — 埋め込み・位置・線形層・RMSNorm

## この章で作るもの

Tensor 自動微分エンジンの上に、トークン埋め込み (token embedding) と位置埋め込み
(position embedding)、チャネルの全結合変換、RMSNorm を実装します。ソース:
`src/main/scala/learnai/transformer/Layers.scala`。

## 離散的な ID から連続的なベクトルへ

トークン ID はカテゴリのラベルです。ID `100` は ID `10` の 10 倍という意味では
ありません。埋め込みテーブル $E\in\mathbb{R}^{V\times C}$ は、各トークンを
$C$ 次元のベクトルに写像します。

```text
token IDs:       [time]
embedding table: [vocabulary,channels]
output:          [time,channels]
```

$$
\boldsymbol{x}_t=E[\text{tokenId}_t]
$$

初期状態のベクトルにはそれ自体の意味はありません。次トークン予測の勾配
(gradient) によって、予測に役立つ配置へと動かされていきます。

## 埋め込みの逆伝播は scatter-add

順伝播 (forward) では行を gather します。逆伝播 (backward) では、各出力の勾配を
元になった行へ scatter します。同じトークン ID が繰り返し現れる場合、勾配は
共有された 1 つのパラメータ行に累積されます。

```text
IDs [2,0,2]
forward:  gather rows 2,0,2
backward: add positions 0 and 2 into row 2
```

これはパラメータ共有のもっとも単純な形です。

## attention には位置情報が必要

位置の信号がないと、self-attention (self-attention) の演算はすべての並べ替えを
区別できません。学習される位置ベクトルを加算します。

$$
\boldsymbol{h}_t^{(0)}=E[x_t]+P[t]
$$

```text
token embeddings:    [time,channels]
position embeddings: [time,channels]
sum:                 [time,channels]
```

同じトークンでも位置が異なれば、異なる隠れ状態から出発します。学習される
絶対位置テーブルは `maximumContextLength` 行しか持たず、それより長い入力は
拒否されます。後の章で扱う RoPE は、クエリとキーのチャネルを回転させることで
位置を導入します。

## 線形層はチャネル軸を変換する

すべての時間方向の行に同じアフィン変換を適用します。

$$
Y=XW+b
$$

```text
X: [time,inputChannels]
W: [inputChannels,outputChannels]
b: [outputChannels]
Y: [time,outputChannels]
```

`addRowVector` はバイアスを行方向に明示的に繰り返します。その逆伝播では、
すべての行の勾配をバイアスに合計します。クエリ、キー、バリュー、出力、
フィードフォワード、ロジットの各射影は、いずれも線形変換です。

## なぜ正規化が必要か

残差 (residual) の加算は、深さ方向に隠れ状態のスケールを変化させることが
あります。極端なスケールは softmax を飽和させ、勾配の流れを損ない、数値的な
挙動を悪化させます。

RMSNorm は平均を引きません。二乗平均平方根 (root mean square) で割り、
チャネルごとに学習されるスケール $g_i$ を適用します。

$$
\operatorname{RMS}(x)=
\sqrt{\frac{1}{C}\sum_i x_i^2+\epsilon}
$$

$$
y_i=g_i\frac{x_i}{\operatorname{RMS}(x)}
$$

`[channels]` の各行は独立に正規化されます。$\epsilon$ はゼロ除算を防ぎます。

## RMSNorm の逆伝播

$r=\operatorname{RMS}(x)$ とし、$g_i$ をスケール適用後の上流勾配とすると、
次のようになります。

$$
\frac{\partial L}{\partial x_i}
=\frac{g_i}{r}
-\frac{x_i}{Cr^3}\sum_j g_jx_j
$$

スケールの勾配は行方向に合計されます。

$$
\frac{\partial L}{\partial \gamma_i}
=\sum_{row}\frac{\partial L}{\partial y_{row,i}}
\frac{x_{row,i}}{r_{row}}
$$

テストでは入力の勾配を有限差分 (finite difference) と比較します。

## パラメータ数

語彙サイズ $V$、コンテキスト長 $T$、チャネル数 $C$ に対して、

- トークン埋め込み: $VC$
- 位置埋め込み: $TC$
- 線形層: $C_{in}C_{out}+C_{out}$
- RMSNorm: $C$

語彙が大きい場合、トークン埋め込みが支配的になることがよくあります。
重み共有 (weight tying) では、その転置を出力分類器として再利用します。

## 実装ウォークスルー

`Embedding` は、形状 `[entries, channels]` の学習可能な重み `Tensor` を所有
します。`apply(indices)` はまずすべての整数を `0 <= id < entries` に対して
検証し、その後 `gatherRows` に委譲します。出力の形状は
`[indices.size, channels]` です。繰り返し現れる ID は意図的に同じ行を選択する
ため、逆伝播ではすべての寄与が 1 つの埋め込みベクトルに累積されます。

3 行 2 チャネルのテーブルを考えます。

```text
E = [[1, 0],
     [0, 1],
     [2, 3]]
ids = [2, 0, 2]
output = [[2,3], [1,0], [2,3]]
```

損失 (loss) が ID `2` の 2 回の出現に対して勾配 `[a,b]` と `[c,d]` を送ると、
2 行目は `[a+c,b+d]` を受け取ります。これはパラメータ共有であって、偶発的な
エイリアシングではありません。

`TokenPositionEmbedding` は、チャネル数が一致していなければならない別々の
トークンテーブルと位置テーブルを所有します。長さ `T` の系列に対しては、
トークン ID でトークン行を、`0 until T` で位置行を gather し、同じ形状の
テンソル同士を加算します。キャッシュ付き推論 (inference) 用のメソッド
`at(tokenId, position)` は、明示された 1 つの位置に対して同じ演算を行います。

`Linear.apply` は `[rows,inputChannels]` を確認し、重み
`[inputChannels,outputChannels]` との matmul を計算し、バイアスのために
`addRowVector` を呼びます。この名前付きブロードキャスト演算が重要なのは、
バイアスの逆伝播ではすべての行にわたる寄与を合計しなければならないからです。

`RmsNorm.apply` は、チャネルごとに 1 つの学習されるスケールを持つ融合された
Tensor 演算に委譲します。epsilon は、行の要素がすべてゼロの場合に逆数平方根を
保護します。

## テストの読み方

参照 (lookup) のテストでは明示的なテーブルを使うため、期待される行はランダム
初期化に依存しません。繰り返しトークンと位置のテストは、同じトークン ID でも
合成後のベクトルが異なりうることを示します。線形層のテストはアフィン変換の
出力を手計算します。RMSNorm のテストは行の二乗平均を確認し、入力とスケール
両方の勾配に有限差分を使います。範囲とコンテキスト上限のテストは、gather の
前にエラーが発生することを保証します。

## デバッグチェックリスト

1. トークン ID、位置 ID、トークン行、位置行を区別する。
2. 値より先にテーブルと出力の形状を確認する。
3. 繰り返し ID の勾配が誤っている場合は、`gatherRows` の累積を調べる。
4. バイアスの勾配が誤っている場合は、行方向の縮約を検証する。
5. RMSNorm が大きな値を出す場合は、学習されるスケールより先に行の二乗平均と
   epsilon を調べる。

## 演習

1. `V=50,000`、`C=4,096` のパラメータ数と fp16 でのバイト数を計算する。
2. 繰り返しトークンの勾配累積を検証する。
3. 位置埋め込みを取り除き、並べ替えた系列の挙動を調べる。
4. バイアス勾配の時間軸方向の合計を導出する。
5. LayerNorm と RMSNorm の式を比較する。
6. epsilon がない場合のゼロ入力の挙動を説明する。

## 完了基準

- トークン ID が順序的ではなくカテゴリ的である理由を説明できる。
- 埋め込み勾配の gather/scatter を説明できる。
- トークン埋め込みと位置埋め込みの形状を述べられる。
- 線形層によるチャネル軸の変化を追跡できる。
- RMSNorm、epsilon、学習されるスケールを説明できる。
- `LayersSuite` が通る。
