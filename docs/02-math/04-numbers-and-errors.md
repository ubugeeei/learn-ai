# 04 — 数、関数、浮動小数点誤差

## この章で作るもの

`Double` の誤差を観察し、許容誤差付き比較と compensated summation を実装します。対象コードは
`src/main/scala/learnai/math/Numerics.scala` です。

## 数学の実数と computer の数は違う

数学の実数 \(\mathbb{R}\) には、整数、有理数、\(\sqrt{2}\)、\(\pi\) などが含まれ、数直線上に
隙間なく存在すると考えます。一方、computer の記憶領域は有限です。Scala の `Double` は
IEEE 754 binary64 形式の 64 bit で、符号、指数、有効数字を表します。

概念的には次の形です。

\[
(-1)^{\text{sign}} \times 1.\text{fraction} \times 2^{\text{exponent}}
\]

2 進数で有限に表せない値は最も近い表現へ丸められます。10 進数の `0.1` もその一つです。

```console
$ nix develop -c sbt 'runMain learnai.math.runFloatingPointLab'
0.1 + 0.2            = 0.30000000000000004
exactly equals 0.3   = false
approximately equal = true
```

これは bug ではなく、有限表現を使う代償です。ニューラルネットワークは膨大な回数の加算と乗算を
行うため、誤差の増幅、overflow、underflow を設計に含めます。

## 絶対誤差と相対誤差

近似値 \(a\) と基準値 \(b\) の絶対誤差は次です。

\[
E_{\mathrm{abs}} = |a - b|
\]

値が \(10^{-8}\) のときの誤差 \(10^{-6}\) は大きい一方、値が \(10^{12}\) なら非常に小さい誤差
です。scale を考慮するため、相対誤差を使います。

\[
E_{\mathrm{rel}} = \frac{|a-b|}{\max(|a|, |b|)}
\]

`approximatelyEqual` は、絶対・相対 tolerance の大きいほうを許容範囲にします。

```scala
val difference = math.abs(left - right)
val scale = math.max(math.abs(left), math.abs(right))
difference <= math.max(absoluteTolerance, relativeTolerance * scale)
```

0 付近では絶対 tolerance、大きな値では相対 tolerance が主に働きます。

## 特別な値

`Double` には通常の有限値以外もあります。

- `Double.PositiveInfinity`: 正の overflow や 0 除算などで現れる
- `Double.NegativeInfinity`: 負の無限大
- `Double.NaN`: `0.0 / 0.0` など、数として定まらない結果

`NaN` は自分自身とも等しくありません。学習中に一つでも混じると、多くの演算を通して全体へ
伝播します。値を生成する境界で `isFinite` を検査するのが、原因に近い場所で失敗させる方法です。

## 足す順番で結果が変わる

数学の実数では加算の結合法則が成り立ちます。

\[
(a+b)+c = a+(b+c)
\]

有限精度では、非常に大きな値へ小さな値を足すと小さな桁が失われます。そのため演算順序が変わる
並列 reduction は、最後の bit まで同じ結果にならない場合があります。

Kahan compensated summation は、加算で失われた下位桁の推定値を次の加算で補正します。

```scala
val corrected = next - compensation
val updated = sum + corrected
compensation = (updated - sum) - corrected
sum = updated
```

万能ではありませんが、異なる magnitude の値を多数足すときに誤差を減らせます。この教材の
`VectorD.sum` はこの実装を使います。

## 数値安定性

数学的に等しい式でも、有限精度での安定性は異なります。たとえば後で softmax を計算するとき、
そのまま `exp(logit)` を計算すると overflow し得ます。全 logit から最大値を引いても確率は
変わらないため、範囲を小さくしてから `exp` を計算します。

数値計算では次の順で考えます。

1. 入力の取り得る範囲は何か
2. 中間値は overflow/underflow しないか
3. 近い値の引き算で有効桁が失われないか
4. 演算回数と誤差はどう増えるか
5. より安定な同値変形がないか

## 演習

1. `1e16 + 1.0 == 1e16` の結果を予想して実行してください。
2. `Vector(1e16, 1.0, 1.0, -1e16)` の順序を変え、二つの sum を比較してください。
3. `Double.MaxValue * 2.0` と `Double.MinPositiveValue / 2.0` を観察してください。
4. relative tolerance を大きくし過ぎると、どんな誤りを見逃すか例を作ってください。

## 完了条件

- 数学の実数と `Double` の違いを説明できる
- 完全一致を使ってよい値と、許容誤差が必要な値を区別できる
- `NaN` が学習計算へ混じる危険を説明できる
- `NumericsSuite` を含む `sbt test` が成功する
