# plamo-3-translate.c

Apple Silicon / macOS 向けの C・Objective-C・Metal・手書き AArch64 アセンブリによる PLaMo 3 Translate の推論実装です。0.8B・2B・8B・31Bの公式BF16重みを直接読み込み、任意で無損失の実行用ファイルを事前作成できます。CLI と API は同じ推論エンジンを使い、既定で Metal GPU を利用します。

[gpt2.s](https://github.com/ArneshBanerjee/gpt2.s) に着想を得た CPU カーネルに、Apple Silicon の共有メモリを使う GPU 実装を加えています。活性値と演算は FP32、重みは元の BF16 で、FP16 への丸めや追加の量子化を行いません。従来のアセンブリだけの CLI は `build/plamo-asm` に残しています。この実行ファイルは Darwin syscall を使い、`nm -u build/plamo-asm` は空です。

## モデルを選ぶ

取得・翻訳・API起動に同じ `--model 0.8b|2b|8b|31b` を指定します。省略時は従来どおり8Bです。環境変数 `PLAMO_MODEL` でも既定値を設定できます。モデルごとに別の静的記述子を生成し、演算カーネルは共通です。

| モデル | BF16ファイル | 合計サイズ | 常駐判定の下限（重み＋2 GiB） | 開発時の保存先 |
| --- | ---: | ---: | ---: | --- |
| 0.8B | 1 | 1.78 GB | 3.66 GiB | `models/0.8b` |
| 2B | 2 | 5.76 GB | 7.36 GiB | `models/2b` |
| 8B | 4 | 17.06 GB | 17.89 GiB | `models/plamo` |
| 31B | 14 | 63.51 GB | 61.15 GiB | `models/31b` |

この下限はCLI・APIの`resident`用です。KV・入力処理・OS用のメモリも必要です。31Bは24 GiBの検証機では完全常駐できませんが、`auto`は空きに応じて一部の重みを保持できます。全モードで重みをディスクに置く必要があります。大きな行列は行ブロックに分割して扱います。

モデルと公式リビジョンは [tools/profiles](tools/profiles) に固定しています。0.8B・2BはAttentionの幅が隠れ層の幅と異なり、31Bは出力ヘッドと入力Embeddingを共有します。設定に従って扱い、学習済み重みや精度は変更しません。

0.8B・2Bの実翻訳・常駐API、8Bの既存ハッシュ一致、31Bの全ヘッダーとローダーを確認しています。[モデル別の検証範囲と再現手順](docs/MODELS.md)を参照してください。

## Nixで利用する（推奨）

Apple SiliconのmacOSと、[Nix](https://nix.dev/install-nix)（`nix-command`・`flakes`を有効化）が必要です。NixがClangとApple SDKを用意するため、この手順ではXcode Command Line Toolsの別途インストールは不要です。Intel MacとLinuxには対応していません。

クローンせずに、公式のモデルを取得して翻訳できます。

```sh
# 初回のみ。2Bの公式BF16重み約5.76 GBと語彙・ライセンスを取得
nix run github:SuzumiyaAoba/plamo-3-translate.c#download-model -- --model 2b --weights

nix run github:SuzumiyaAoba/plamo-3-translate.c -- --model 2b -j 'Hello.'
nix run github:SuzumiyaAoba/plamo-3-translate.c -- --model 2b -e 'こんにちは。'
nix run github:SuzumiyaAoba/plamo-3-translate.c -- --help
```

既定のデータルートは `${XDG_DATA_HOME:-$HOME/.local/share}/plamo-3-translate-SIZE` です。2Bの保存先はその下の `models/2b`、8Bは従来どおり `models/plamo` です。上表の重みサイズに加え、付属ファイル用の空き容量が必要です。重み・語彙はGitリポジトリ、Nixパッケージ、Nix storeに含めません。Nixのビルドではモデルの取得も推論も実行しません。

保存場所を変える場合、両コマンドに同じ `PLAMO_DATA_DIR` を指定します。以下の例では `$HOME/plamo-data/models/plamo` に保存されます。

```sh
export PLAMO_DATA_DIR="$HOME/plamo-data"
nix run github:SuzumiyaAoba/plamo-3-translate.c#download-model -- --weights
nix run github:SuzumiyaAoba/plamo-3-translate.c -- -j 'Hello.'
```

新規ダウンロードは固定リビジョンのサイズ・SHA-256を検証してから確定します。中断時は再実行すると完了済みファイルを再利用します。取得済み重みもハッシュ検証するには `#download-model -- --weights --verify` を使います。モデルの利用条件は[公式ライセンス](https://huggingface.co/pfnet/plamo-3-translate-8b/blob/19ecab3fb31424ee937b7503730318b50e55771c/LICENSE/ja)を確認してください。

ローカルで開発する場合:

```sh
git clone https://github.com/SuzumiyaAoba/plamo-3-translate.c.git
cd plamo-3-translate.c
nix flake check                  # モデルなしでビルド・テスト
nix build                       # result/bin/plamo、result/bin/plamo-download
nix develop                     # 固定された開発環境に入る
make all test test-tools
python3 tools/download.py --weights  # 開発時は ./models/plamo に取得
./build/plamo -j 'Hello.'
```

`nix develop` は、未指定なら `PLAMO_DATA_DIR` を作業ディレクトリに設定します。既存の `./models/plamo` をNixパッケージから使うこともできます。

```sh
PLAMO_DATA_DIR="$PWD" nix run . -- -j 'Hello.'
```

`flake.lock` がNixpkgsを固定します。`nix flake check` は合成データによる演算テスト、ダウンローダー・起動コマンドのオフラインテスト、生成ソースの再現性、従来のアセンブリバイナリの未解決シンボルがないことを検証します。Nixがflakesを無効としてエラーにする場合は、`~/.config/nix/nix.conf` に `experimental-features = nix-command flakes` を設定してください。

単発CLIのNix起動コマンドは保存先を選んでネイティブ本体を実行するシェルです。このCLIではPythonを推論中に呼び出しません。モデル別の本体を `result/libexec/plamo-SIZE`、従来のアセンブリ CLI を `result/libexec/plamo-asm-SIZE` にインストールします。`result/libexec/plamo` は8Bです。

## 常駐API

```sh
nix run github:SuzumiyaAoba/plamo-3-translate.c#serve -- --model 2b --memory auto --port 8000
```

別ターミナルから:

```sh
curl http://127.0.0.1:8000/health
curl -N http://127.0.0.1:8000/translate \
  -H 'Content-Type: application/json' \
  -d '{"text":"Hello.","source_lang":"en","target_lang":"ja","stream":true}'
```

1つのワーカーがモデルと語彙を保持します。リクエスト間ではKV・RoPEの状態をリセットし、保持済みの固定翻訳テンプレートと入力トークンが一致すれば、その接頭12トークンのKVだけを復元します。`auto`は起動時の空きメモリの推定値から2 GiBの余裕を引いた量を上限に、実際に使う元BF16の行列を保持可能な範囲まで起動時に先読みします。先読みが終わるまでは`/health`が`loading`を返し、完了してから`ready`になります。事前作成ファイルを利用しない場合、残りは最大64 MiBの固定領域へ都度読み込みます。大きな行列も行単位で分割し、空きが減れば保持する重みを解放し、回復すれば起動時の上限内で保持量を増やします。全重みが収まらなくても起動でき、保持できない範囲の先読みI/Oは行いません。単発CLIの`auto`は最初の翻訳中に保持可能な行列を準備し、入力全体をGPUで処理できるようにします。

完全常駐を要求する場合は`--memory resident`を指定してください。CPUでは重みと2 GiBの余裕分が必要です（2Bは約7.90 GB、8Bは約19.21 GB）。GPUは収まりきらない場合に元BF16の全ビットを保持する圧縮キャッシュを選び、実際の保持領域と余裕分が収まるかを確認します。8Bの計算用行列では約16.18 GBを約12.30 GBに保持できました。

CLI・API は既定で利用可能なMetal GPUを使い、初期化できない場合はCPUへ戻ります。`--backend metal`でGPUを必須にでき、`--backend metal-fp32`も同じFP32経路を使います。`--backend cpu`はCPUアセンブリを使います。必要な行列が専用バッファまたは実行用ファイルで参照可能なら、入力処理全体と生成の各1トークン分について、全層・Attention・KV更新・RoPE・MLP・出力ヘッドを一つのGPUコマンドにまとめます。重みは元BF16、活性値・累積演算はFP32です。APIでは小さな正規化重み、RoPE表、GPU作業領域、英日・日英の固定接頭KVを起動時に準備します。固定接頭の準備は起動時間に含まれ、本文は要求ごとに計算します。[精度・共有メモリ・検証結果](docs/METAL.md)を参照してください。

実際のバックエンド・GPU名と、メモリモード（`auto`は`adaptive`）・領域の上限・保持中の重み・読み込み領域のバイト数は`/health`に表示します。`load_seconds`は先読みを含む起動時間、`preload_seconds`は追加の準備時間です。`memory.preloaded_weight_bytes`は起動時の先読み量、`cached_weight_bytes`は元BF16換算の保持量、`cached_storage_bytes`は実際のキャッシュ領域です。I/Oを減らせる量は保持できる重みに依存します。APIの制約、ストリーミング、計測結果は[常駐APIの説明](docs/API.md)を参照してください。APIの制御層はC・Objective-CとPython標準ライブラリ、CPUカーネルはアセンブリ、GPUカーネルはMetalです。

ready後の初回応答も最適化しています。M2・24 GiBの短文では、最初のトークンまで0.8Bが0.093→0.029秒、2Bが0.217→0.100秒、8Bが0.657→0.404秒でした（3回の中央値、前後で保持形式・保持量を統一）。入力処理全体のGPU化と固定接頭KVの再利用による改善で、固定接頭の計算は起動時間に含みます。[条件・長文・精度検証](docs/METAL.md)を参照してください。

CPUバックエンドの大きな行列演算は、既定で性能コア数に合わせて並列実行します。`--threads 4`でスレッド数を指定でき、`--threads 1`で逐次実行になります。実際の数は`/health`の`threads`に表示します。2Bの完全常駐では、従来の1スレッドに対し入力21トークンの処理が1.52秒→0.42秒、生成1ステップが0.24秒→0.11秒でした（3回の中央値）。CPU最適化前後ではBF16重みと演算結果のハッシュが一致しています。[LM Studioとの速度差・比較条件](docs/OPTIMIZATION.md#lm-studioとの速度差とcpu並列化)を参照してください。

8Bは[事前作成した実行用重みファイル](docs/WEIGHT_CACHE.md)をGPUから直接参照し、0.8Bによる候補作成と8Bの一括検証を組み合わせます。M2・24 GiBの通常設定の実APIで、英→日 **12.322 tokens/s**、日→英 **10.637 tokens/s**（各3回の中央値）を確認しました。ready直後から最大64生成トークンを測り、全6要求で10 tokens/sを超え、SSE受信時刻の速度と従来の8Bトークン列・翻訳文も一致しました。8B本体は元のBF16重み・FP32演算を維持し、補助モデルだけをQ4で保持します。通常生成とのFP32の丸め差はあります。[導入・精度検証](docs/METAL.md#8bの候補検証による生成)と[測定データ](docs/benchmarks/speculative.json)を参照してください。補助重みは`plamo-download --model 0.8b --weights`で用意でき、無効化は`PLAMO_SPECULATIVE=0`です。

## Nixを使わずに実行

macOS、Apple Silicon、Xcode Command Line Tools が必要です。既定の8Bは4分割で約17 GBあります。検証機は24 GiB RAMです。CLI も Metal GPU・適応型重みキャッシュを使い、CPU を選ぶ場合は性能コアで並列計算します。入力処理は複数トークンの射影をまとめる prefill を使います。重みの読み込みが性能に影響します。

```sh
python3 tools/download.py --weights
make all
./build/plamo 'こんにちは😀'       # トークン ID の表示
./build/plamo -g 'raw prompt'      # テンプレートなしの生成
./build/plamo -j 'Hello.'          # 英語から日本語
./build/plamo -e 'こんにちは。'    # 日本語から英語
./build/plamo --backend metal --memory auto -j 'Hello.'
./build/plamo --backend cpu --threads 4 -j 'Hello.'
```

生成は greedy、UTF-8 テキスト出力、既定256トークンです。`--max-tokens 1..4096`で上限を指定できます。終了 ID は公式 generation_config と同じ1・16です。翻訳は基本の PFML3 プロンプトを使います。入力はプロンプト込み4096 Unicode 符号位置までです。用語集・文体・例示などの追加テンプレート機能は CLI にありません。翻訳入力への予約特殊トークン混入は現在拒否しません。

別サイズの例:

```sh
python3 tools/download.py --model 0.8b --weights
make MODEL=0.8b all
./build/0.8b/plamo -j 'Hello.'
python3 tools/server.py --model 0.8b --memory auto
# 全4モデルのバイナリを生成（重み不要）
make models
```

`MODEL=2b`・`MODEL=31b`も同様です。8Bの実行ファイルは互換性のため `build/plamo`、他は `build/SIZE/plamo` です。各モデルの重み・語彙は上表の保存先へ取得し、公式リビジョンを固定しています。ダウンロードの Python は推論に使いません。

## 実装

- BF16 → FP32 の NEON 行列ベクトル演算、オフセット付き RMSNorm、指数関数・softmax・SwiGLU
- 結合 QKV 射影、Q/K ヘッド正規化、GQA、出力射影
- 回転前キーを保持する KV キャッシュ、スライディング時の古いトークン除去
- ローカル／YaRN RoPE、キャッシュのローカル位置での回転
- Attention・MLP の正規化と残差、16 / 24 / 32 / 64 Decoder層、最終正規化とlogits
- モデル別テンソルのファイル横断検索、形状照合、登録、31Bの重み共有
- 公式 Unigram 語彙の読み込み、Unicode Trie、整数 DP、バイトフォールバック
- 特殊トークン・連続文字の境界前処理、UTF-8 復号

キャッシュと作業領域は匿名 mmap で仮想領域を確保します。全重みの FP32 コピーは作りません。RoPE の周波数などの固定モデル定数と記述子は `tools/profiles/SIZE/config.json` から生成しています。`docs/config.json` は従来の8B設定です。8Bの起動時の圧縮を省くには、`python3 tools/prepare_weights.py --model 8b`で[実行用重みファイル](docs/WEIGHT_CACHE.md)を事前作成できます。元の全BF16ビットを保持します。

## 検証

```sh
make test
make build/test-model-open
./build/test-model-open
python3 tools/verify_weights.py
python3 tools/verify_segments.py
python3 tools/verify_boundaries.py
nm -u build/plamo-asm
```

C はCLI・APIのモデル常駐・状態管理と演算テストに使います。8Bの公式語彙107,520行・323テンソルの読み込みを検証済みです。4モデルすべてで、合成BF16ファイルを使った実ローダーの形状照合・初期化失敗後の再試行・再初期化をテストします。31Bの全14ファイル・642テンソルは公式ヘッダーとも照合済みですが、31Bの実翻訳はメモリ制約により未検証です。区間 Unigram は独立参照210ケース、境界検出は公式正規表現と各2035ケースで比較しています。

公式コードとの直接比較には別途 PyTorch、Transformers 4.57.1、Numba、safetensors が必要です。

```sh
HF_HOME=/tmp/plamo-hf .venv/bin/python tools/verify_official_tokenizer.py
HF_HOME=/tmp/plamo-hf .venv/bin/python tools/reference_first.py Hello
HF_HOME=/tmp/plamo-hf .venv/bin/python tools/reference_first.py --source en --target ja 'Hello.'
```

独立 CLI のトークナイザーは公式と33ケースで一致しました。入力 `Hello` の先頭生成 ID は公式 FP32 参照・アセンブリとも7272です。全32層の中間 RMS も近い値ですが、要素ごとの一致や長文生成・翻訳の一致を証明するものではありません。公式参照はメモリ節約のため1層ずつ FP32 で読み込む検証プログラムです。

## 残る検証・制約

- `Hello.` → 「こんにちは。」を公式 FP32 参照と照合済み。幅広い文章・言語対での精度比較は未実施
- BF16 で演算する公式実行との完全一致は未確認
- prefill は4/8トークン単位の行列積を使用。出力の加算順序は維持
- 長いコンテキスト、スライディング境界をまたぐ実モデルの結果は未検証
- safetensors は検索対象を発見した時点で終了し、ファイル全体の重複領域などは検証しない
- 出力は部分書き込み・EINTR を再試行する共通 Darwin syscall 関数を使用

公式 FP32 参照で `Hello.` の英日翻訳は ID 51871（「こんにちは。」）、続いて終了 ID 16となることを確認しました。アセンブリも「こんにちは。」を出力して終了し、この例の翻訳全体が一致しました。入力処理では最終トークン以前の logits 計算を省略します。

確認済みの実行例:

```text
$ ./build/plamo -j 'Hello.'
こんにちは。
```

初回読み込みと CPU 計算に時間がかかります。処理速度の目標や GPU と同等の速度を保証するものではありません。

## 最適化結果

8Bで完全一致を維持した実モデルの測定（Apple Silicon / 24 GiB、最適化後2回の中央値）:

| 項目 | 最適化前 | 最適化後 |
| --- | ---: | ---: |
| 入力処理・21トークン | 50.28秒 | 9.09秒 |
| 生成1ステップ | 34.79秒 | 5.22秒 |
| 上記の合計 | 85.08秒 | 14.31秒（約5.95倍） |
| ピーク常駐メモリ | 8.47 GB | 0.90 GB |

最大1 GiBの元バイトまとめ読み、8行同時の行列ベクトル積、4/8トークンの行列積、Attention・活性化・キャッシュの SIMD 化、回転済みキーの再利用、語彙の直接索引を実装しています。トークン化コマンドは中央値0.796秒→0.0326秒でした。

実モデルの全logits・最終状態892,928バイトを最適化前と直接比較し、一致を確認しました。時間・常駐量は入力長や OS のキャッシュで変動します。手順・比較条件・採否の記録は [最適化の詳細](docs/OPTIMIZATION.md) にあります。

```sh
python3 tools/benchmark.py --repeats 3
```

## ライセンス

このリポジトリの推論コード・Nix定義・テストは [MIT License](LICENSE) です。モデルは同梱しておらず、別途取得するモデルには PLaMo Community License が適用されます。公開設定に基づく構造メタデータと、文書中のモデル出力例も区別して扱います。[第三者のモデル・参考実装](THIRD_PARTY_NOTICES.md)を参照してください。

Built with PLaMo.
