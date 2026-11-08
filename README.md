# plamo-3-translate-8b.s

Apple Silicon / macOS 向けの手書き AArch64 アセンブリによる PLaMo-3-Translate-8B 推論実装です。公式 BF16 重みを直接読み込み、英日翻訳の実行と公式 FP32 参照との一致を確認しています。

[gpt2.s](https://github.com/ArneshBanerjee/gpt2.s) と同様に、単発CLIの推論本体は C / libc / Python / Metal / Accelerate 呼び出しを使いません。Darwin syscall で公式 safetensors と tokenizer.jsonl を変換せず読みます。ヘッダーは mmap、大きな BF16 行列は最大1 GiBの再利用領域へのまとめ読みを使います。libSystem は macOS ローダーのためにリンクしますが、`nm -u build/plamo` は空です。

## Nixで利用する（推奨）

Apple SiliconのmacOSと、[Nix](https://nix.dev/install-nix)（`nix-command`・`flakes`を有効化）が必要です。NixがClangとApple SDKを用意するため、この手順ではXcode Command Line Toolsの別途インストールは不要です。Intel MacとLinuxには対応していません。

クローンせずに、公式のモデルを取得して翻訳できます。

```sh
# 初回のみ。公式の元BF16重み約17 GBと語彙・ライセンスを取得
nix run github:SuzumiyaAoba/plamo-3-translate-8b.s#download-model -- --weights

nix run github:SuzumiyaAoba/plamo-3-translate-8b.s -- -j 'Hello.'
nix run github:SuzumiyaAoba/plamo-3-translate-8b.s -- -e 'こんにちは。'
nix run github:SuzumiyaAoba/plamo-3-translate-8b.s -- --help
```

モデル保存先は `${XDG_DATA_HOME:-$HOME/.local/share}/plamo-3-translate-8b/models/plamo` です。モデル用に約18 GB以上の空き容量が必要です。重み・語彙はGitリポジトリ、Nixパッケージ、Nix storeに含めません。Nixのビルドではモデルの取得も推論も実行しません。

保存場所を変える場合、両コマンドに同じ `PLAMO_DATA_DIR` を指定します。以下の例では `$HOME/plamo-data/models/plamo` に保存されます。

```sh
export PLAMO_DATA_DIR="$HOME/plamo-data"
nix run github:SuzumiyaAoba/plamo-3-translate-8b.s#download-model -- --weights
nix run github:SuzumiyaAoba/plamo-3-translate-8b.s -- -j 'Hello.'
```

新規ダウンロードは固定リビジョンのサイズ・SHA-256を検証してから確定します。中断時は再実行すると完了済みファイルを再利用します。取得済み重みもハッシュ検証するには `#download-model -- --weights --verify` を使います。モデルの利用条件は[公式ライセンス](https://huggingface.co/pfnet/plamo-3-translate-8b/blob/19ecab3fb31424ee937b7503730318b50e55771c/LICENSE/ja)を確認してください。

ローカルで開発する場合:

```sh
git clone https://github.com/SuzumiyaAoba/plamo-3-translate-8b.s.git
cd plamo-3-translate-8b.s
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

`flake.lock` がNixpkgsを固定します。`nix flake check` は合成データによる演算テスト、ダウンローダー・起動コマンドのオフラインテスト、生成ソースの再現性、推論バイナリの未解決シンボルがないことを検証します。Nixがflakesを無効としてエラーにする場合は、`~/.config/nix/nix.conf` に `experimental-features = nix-command flakes` を設定してください。

単発CLIのNix起動コマンドは保存先を選んでアセンブリ本体を実行するシェルです。このCLIではPythonはモデル取得専用で、推論中には呼び出しません。アセンブリ本体は `result/libexec/plamo` にもインストールされます。

## 常駐API

```sh
nix run github:SuzumiyaAoba/plamo-3-translate-8b.s#serve -- --memory auto --port 8000
```

別ターミナルから:

```sh
curl http://127.0.0.1:8000/health
curl -N http://127.0.0.1:8000/translate \
  -H 'Content-Type: application/json' \
  -d '{"text":"Hello.","source_lang":"en","target_lang":"ja","stream":true}'
```

1つのワーカーがモデルと語彙を保持し、リクエスト間ではKV・RoPEの状態をリセットします。`auto`は十分な余裕があれば元のBF16重み約17 GBを読み取り専用の匿名メモリへ先に読み込み、不足する場合は従来の省メモリI/Oへ戻ります。完全常駐を要求する場合は`--memory resident`を指定してください。重みと2 GiBの余裕分、計約19.2 GBの再利用可能メモリが起動時の目安です。

空き不足では常駐APIだけで生成が速くなるわけではありません。実際のモードと理由は`/health`に表示します。APIの制約、ストリーミング、計測結果は[常駐APIの説明](docs/API.md)を参照してください。APIの制御層はCとPython標準ライブラリ、推論カーネルは既存のアセンブリです。

## Nixを使わずに実行

macOS、Apple Silicon、Xcode Command Line Tools が必要です。重みは4分割で約17 GBあります。検証機は24 GiB RAMです。FP32 の中間値と BF16 の元重みを使う、CPU 1スレッドの実装です。入力処理は複数トークンの射影をまとめる prefill を使います。重みの読み込みが性能に影響します。

```sh
python3 tools/download.py --weights
make all
./build/plamo 'こんにちは😀'       # トークン ID の表示
./build/plamo -g 'raw prompt'      # テンプレートなしの生成
./build/plamo -j 'Hello.'          # 英語から日本語
./build/plamo -e 'こんにちは。'    # 日本語から英語
```

生成は greedy、UTF-8 テキスト出力、最大256トークンです。終了 ID は公式 generation_config と同じ1・16です。翻訳は基本の PFML3 プロンプトを使います。入力はプロンプト込み4096 Unicode 符号位置までです。用語集・文体・例示などの追加テンプレート機能は CLI にありません。翻訳入力への予約特殊トークン混入は現在拒否しません。

重み・語彙は `models/plamo/` に保存し、リビジョン `19ecab3fb31424ee937b7503730318b50e55771c` に固定しています。ダウンロードの Python は推論に使いません。

## 実装

- BF16 → FP32 の NEON 行列ベクトル演算、オフセット付き RMSNorm、指数関数・softmax・SwiGLU
- 結合 QKV 射影、Q/K ヘッド正規化、GQA、出力射影
- 回転前キーを保持する KV キャッシュ、スライディング時の古いトークン除去
- ローカル／YaRN RoPE、キャッシュのローカル位置での回転
- Attention・MLP の正規化と残差、32 Decoder 層、最終正規化と logits
- 323テンソルの分割ファイル横断検索、形状照合、登録
- 公式 Unigram 語彙の読み込み、Unicode Trie、整数 DP、バイトフォールバック
- 特殊トークン・連続文字の境界前処理、UTF-8 復号

キャッシュと作業領域は匿名 mmap で仮想領域を確保します。全重みの FP32 コピーは作りません。RoPE の周波数などの固定モデル定数と記述子は `docs/config.json` から生成しています。モデル重みの事前変換はありません。

## 検証

```sh
make test
make build/test-model-open
./build/test-model-open
python3 tools/verify_weights.py
python3 tools/verify_segments.py
python3 tools/verify_boundaries.py
nm -u build/plamo
```

C は演算テストのハーネスと、API用のモデル常駐・状態管理に使います。単発CLIの推論はアセンブリだけです。公式語彙107,520行・323テンソルの読み込みを検証済みです。区間 Unigram は独立参照210ケース、境界検出は公式正規表現と各2035ケースで比較しています。

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

完全一致を維持した実モデルの測定（Apple Silicon / 24 GiB、最適化後2回の中央値）:

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
