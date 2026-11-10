# 実行用重みファイルの事前作成

起動時に作るGPU用バッファを、一度だけ別ファイルへ保存できます。8Bで通常行う「元BF16の読み込み→指数の解析→無損失圧縮」を、次回から「保存済みの圧縮ブロックの読み込み」に置き換えます。元のBF16全16ビットを保ち、演算は従来と同じFP32です。

## 作成と利用

```sh
# Nixパッケージ。取得済みの8Bから作成
./result/bin/plamo-prepare-weights --model 8b
# または nix run .#prepare-weights -- --model 8b

# リポジトリ内のモデルを使う場合
python3 tools/prepare_weights.py --model 8b
```

既定の保存先はモデルのディレクトリ内の`metal-cache-v1.bin`です。CLI・常駐APIのMetalバックエンドが、次回の起動時から自動で検出します。実際の保存形式が一致するブロックだけを使い、従来のメモリ予算内で保持します。8Bの計算用行列16,181,624,832バイトに対して、作成したファイルは12,295,487,488バイトでした。今回の初回変換は約36秒で、公式重みのSHA-256検証と全BF16値の復元照合を含みます。

元のsafetensorsはEmbedding・小さな重み・キャッシュ未収録部分の読み込みにも使うため残します。変換ファイルは追加のディスク容量を使います。変換は一時ファイルへ行い、完了時に置き換えます。既に起動中のワーカーは、開いている古いファイルを引き続き使えます。

保存先や利用の切り替えは次の通りです。

```sh
python3 tools/prepare_weights.py --model 8b --output /path/to/8b-cache.bin
PLAMO_METAL_CACHE=/path/to/8b-cache.bin python3 tools/server.py --model 8b --backend metal
# 同じバイナリで元の読み込み経路へ戻す
PLAMO_METAL_CACHE=off python3 tools/server.py --model 8b --backend metal
```

`PLAMO_WEIGHT_STORAGE=auto|raw|lossless`による保存形式の選択は従来通りです。通常BF16を選ぶ環境では、無損失圧縮のファイルは使いません。2Bなどの比較用に`prepare_weights.py --storage raw`も使えますが、今回の2Bでは起動短縮を確認できず、通常形式のモデルに対する事前変換は推奨していません。CPU・`--memory stream`は元の経路を使います。

無効なファイル、異なるリビジョン、形状・形式の合わないブロックは元の読み込みへ戻ります。理由は`compute.note`、実際の利用は次の累積値で確認できます。

| 診断値 | 意味 |
| --- | --- |
| `compute.disk_cache_hits` | 保存済みブロックを読み込んだ回数 |
| `compute.disk_cache_weight_bytes` | その元BF16換算の合計バイト数 |
| `compute.disk_cache_storage_bytes` | ページ境界の余白を含む読み込み量 |

これらは現在の保持量ではありません。メモリ圧後の再読み込みも加算します。現在の保持量は従来の`memory.cached_weight_bytes`・`cached_storage_bytes`です。明示的なファイル読み込みは、従来の`weight_io`にも含めます。

## 形式と検証

先頭に形式バージョン・元モデルのリビジョン・元ファイルのサイズ・ブロック一覧を保存します。計算用の行列は推論順に並べ、最大64 MiB相当の元BF16を単位として、各保存領域を16 KiB境界へそろえます。圧縮ブロックは既存の512値単位の形式そのもので、GPUカーネルは変更していません。

作成前に公式の固定SHA-256と元ファイルを照合し、圧縮時は各ブロックをGPUで復元して全BF16値をビット単位で比較します。起動時はファイル全体のハッシュを再計算せず、リビジョン・元ファイルサイズ・範囲・圧縮ヘッダーの全オフセットを検証します。読み込み経路は既存のMetal共有バッファへの一括読み込みです。mmapしたファイルをGPUへ直接渡す案は、小規模試験では動作しましたが、2Bの起動が遅くなったため採用していません。

保持できない重みの都度読み込みは、従来の元BF16経路です。事前変換した圧縮ブロックを都度読む案には、2回目の応答が遅くなる条件があったため採用していません。

```sh
make test test-tools
MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1 ./build/test-metal --require-gpu
python3 tools/verify_disk_cache.py --model 2b --storage raw --file /path/to/2b-cache.bin
python3 tools/benchmark_weight_cache.py --model 8b --storage lossless --file /path/to/8b-cache.bin --budget-mib 8192 --repeats 3
```

## 測定

Apple M2・24 GiB・macOS 26.5.1で、同じワーカーのファイル利用を無効・有効にし、実行順を交互にして3回比較しました。ウォームアップ要求を送らず、ready後に同じ要求を2回送ります。前後で保持量・保存形式が一致する組だけから中央値を算出します。8Bは比較専用ワーカーの重み保持とステージングの上限を8 GiBにそろえた部分保持条件で、2Bは全計算用重みを保持しています。OSのファイルキャッシュ・他アプリの状態は制御していません。

| 条件 | 元の読み込み | 事前作成ファイル |
| --- | ---: | ---: |
| 8B・起動 | 6.011秒 | 3.608秒 |
| 8B・ready後の初回トークン | 2.828秒 | 1.794秒 |
| 8B・2回目の初回トークン | 1.676秒 | 1.735秒 |
| 2B・起動 | 1.403秒 | 2.265秒 |
| 2B・ready後の初回トークン | 0.101秒 | 0.099秒 |
| 2B・2回目の初回トークン | 0.096秒 | 0.098秒 |

8Bの起動は約40%短縮しました。2回目の応答は改善しておらず、2Bの通常BF16では起動が遅くなりました。8B・2Bの各10要求で全状態・logitsのハッシュ、トークン列、テキストが一致しました（8Bは最大2生成トークン、2Bは最大12）。

各回、保持量、初回・2回目の応答、検証結果、ソースとバイナリのハッシュは[測定データ](benchmarks/weight-cache.json)に記録します。ファイル形式の変更だけで、保持済み行列の演算速度が上がるものではありません。起動とready後の応答は別々に判断してください。
