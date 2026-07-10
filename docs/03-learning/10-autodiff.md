# 10 — reverse-mode 自動微分

## この章で作るもの

scalar 演算の computation graph を記録し、出力から全 parameter への gradient を一回の逆向き
走査で求める自動微分 engine `Value` を実装します。対象コードは
`src/main/scala/learnai/autodiff/Value.scala` です。

## 数値式を graph として見る

次の計算を考えます。

\[
x=2,\quad y=-3,\quad z=xy+x^2
\]

program は通常、途中の演算を終えると結果だけを残します。自動微分では、各演算を node、値の
依存関係を edge として記録します。

```text
x ----(*)----+
 \   /       |
  \ y        (+)---- z
   \         |
    --pow(2)-+
```

`Value` は各 node に次を保持します。

- `data`: forward 計算の結果
- `gradient`: 最終出力をこの node で微分した値
- `operation`: この node を作った演算
- `previous`: 直接の入力 node
- `backwardRule`: 出力側 gradient を入力へ配る局所微分

## forward mode と reverse mode

微分の計算方向には大きく二つあります。

- forward mode: 一つの入力から全出力への微分を伝える
- reverse mode: 一つの出力から全入力への微分を伝える

neural network は非常に多くの parameter から一つの scalar loss を作ります。入力が多数、出力が
一つなので、一回の逆向き走査で全 parameter gradient を得る reverse mode が適します。

## 局所微分と上流 gradient

ある node \(c=f(a,b)\) に、最終 loss \(L\) からの gradient \(\partial L/\partial c\) が届いたと
します。連鎖律で入力へ伝えます。

\[
\frac{\partial L}{\partial a}
=\frac{\partial L}{\partial c}\frac{\partial c}{\partial a}
\]

乗算 \(c=ab\) の局所微分は \(\partial c/\partial a=b\)、\(\partial c/\partial b=a\) です。

```scala
output.backwardRule = () =>
  accumulateGradient(other.data * output.gradient)
  other.accumulateGradient(data * output.gradient)
```

`output.gradient` が上流から届いた gradient、`other.data` と `data` が局所微分です。

## gradient は代入でなく加算する

\(z=x^2+x\) では \(x\) から出力へ二つの経路があります。

\[
\frac{dz}{dx}=2x+1
\]

一つ目の経路から `2x`、二つ目から `1` が届きます。後から届いた値で上書きすると片方を失います。
そこで `accumulateGradient` は `+=` の意味で加算します。

parameter sharing、residual connection、同じ token embedding の複数回参照がある LLM では、
この gradient accumulation が必須です。

## なぜ topological order が必要か

ある node の backward rule は、その node への全経路の gradient が集まった後に一度実行する必要が
あります。依存元が先、出力が後になる topological order を作り、逆順に走査します。

```text
forward order:  leaves -> intermediate -> loss
backward order: loss   -> intermediate -> leaves
```

同じ `Value` が複数経路で現れるため、通常の値の等価性でなく object identity を使って訪問済みを
記録します。

最終出力 \(L\) 自身の微分は 1 です。

\[
\frac{\partial L}{\partial L}=1
\]

これを seed として逆伝播を始めます。

## 演算ごとの局所微分

`Value` は次を実装します。

| forward | 入力への局所微分 |
| --- | --- |
| \(a+b\) | \(1, 1\) |
| \(ab\) | \(b, a\) |
| \(a^r\) | \(ra^{r-1}\) |
| \(e^a\) | \(e^a\) |
| \(\log a\) | \(1/a\) |
| \(\tanh a\) | \(1-\tanh^2a\) |
| \(\operatorname{ReLU}(a)\) | \(a>0\) なら 1、それ以外 0 |

引き算と割り算は、既存演算の合成として実装します。

\[
a-b=a+(-1)b,\qquad \frac{a}{b}=ab^{-1}
\]

## parameter update と graph の寿命

演算 node の `data` は forward 時点の snapshot です。正しい一 step は次の順です。

1. 現在の parameter から新しい graph を作る
2. scalar loss で `backward()` を呼ぶ
3. parameter の `gradient` を使って update する
4. 古い graph を捨て、次の forward で作り直す

古い graph の途中値は parameter update 後も変わりません。graph を再利用してはいけません。
実用 framework も通常、dynamic graph を backward 後に解放します。

## gradient check

自動微分の test は、同じ関数を raw `Double` と `Value` で作り、central difference と比較します。

```scala
val numeric = Calculus.derivative(rawFunction, at = x.data)
output.backward()
Assert.close(x.gradient, numeric, tolerance = 1e-8)
```

解析的な期待値、数値微分、自動微分という独立した三つの見方を使うと、局所微分や graph 順序の
bug を見つけやすくなります。

## 今回の単純化

- scalar node 一つにつき JVM object を作るため遅い
- recursive な topological sort は巨大 graph で stack を使い過ぎる
- higher-order derivative を保持しない
- in-place 演算や並列 backward を扱わない
- graph visualization と profiler がない

次の `Tensor` engine では一つの node に多数の数を持たせます。実用 framework はさらに演算 fusion、
device memory、kernel scheduling を管理します。

## 演習

1. \(z=(x+2)^3\) を `Value` で作り、手計算と gradient を比較してください。
2. `sigmoid` を既存の `exp` と四則演算から合成してください。
3. `sin` 演算と backward rule を追加し、数値微分で検査してください。
4. `accumulateGradient` を一時的に代入へ変え、共有 node の test がどう失敗するか確認してください。
5. graph の各 node と edge を DOT 形式で出力する方法を設計してください。

## 完了条件

- computation graph の node と edge を式へ対応させられる
- reverse mode が多数入力・一出力に向く理由を説明できる
- backward rule を上流 gradient と局所微分に分けて説明できる
- gradient を加算する必要がある式を一つ作れる
- parameter update 後に graph を作り直す理由を説明できる
- `ValueSuite` が成功する
