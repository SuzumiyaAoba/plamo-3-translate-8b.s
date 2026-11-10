# 常駐翻訳API

モデルと語彙を一度読み込んだワーカーを維持し、複数リクエストで再利用します。翻訳を決定する本体の重み・モデル構造・greedy生成は変更していません。リクエスト間ではKVキャッシュの件数とRoPEの状態をリセットし、前の文章を次へ持ち越しません。

## 起動

公式モデルを取得済みなら、次で起動できます。

```sh
nix run github:SuzumiyaAoba/plamo-3-translate.c#serve -- --model 2b --memory auto --port 8000
```

初回の取得は `nix run github:SuzumiyaAoba/plamo-3-translate.c#download-model -- --model 2b --weights` です。`--model`には`0.8b`・`2b`・`8b`・`31b`を指定できます。省略時は`PLAMO_MODEL`または8Bです。1サーバーにつき1つの翻訳モデルを保持し、変更時は再起動します。8Bの候補作成には補助0.8Bワーカーが加わる場合があります。データ保存先はCLIと同じ `PLAMO_DATA_DIR` を使います。開発時の `./models/plamo` を利用する場合:

```sh
PLAMO_DATA_DIR="$PWD" nix run .#serve -- --memory auto
# または
make all
python3 tools/server.py --data-dir "$PWD" --memory auto
```

別のターミナルから確認します。読み込み中・起動失敗は503、利用可能なら200です。

```sh
curl http://127.0.0.1:8000/health
```

`status`、`busy`、`model`、公式リポジトリ・固定リビジョン、`worker_pid`、`threads`、読み込み時間、実際のメモリモード、空きの推定値、フォールバックの理由が返ります。`memory.max_budget_bytes`は起動時に決めた絶対上限、`budget_bytes`は現在有効な重み領域の上限です。`cached_weight_bytes`は元BF16換算の保持量、`cached_storage_bytes`は実際の保持領域、`staging_bytes`は都度読み込み領域のバイト数です。GPUの無損失圧縮では前者が後者を上回ります。`cache_hits`・`cache_hit_bytes`・`cache_misses`・`resizes`も表示します。メモリ情報は起動時・入力処理開始時・翻訳終了時の最新通知を表示し、待機中の調整は次の通知で反映します。起動中でもHTTPの状態確認は応答します。`load_seconds`は語彙・モデル初期化と先読みを含む時間、`preload_seconds`は追加の準備時間です。先読み・GPUページ準備に加え、小さな正規化重み、CPUと同じ計算で作るRoPE表、GPU作業領域も起動時に準備します。`resident`で従来から行う全重み読み込みも`load_seconds`に含みます。`memory.preloaded_weight_bytes`は起動完了時点の先読み量で、後のメモリ圧による解放後も起動時の値を表示します。

既定の待ち受けは127.0.0.1です。別のホストへbindする場合は`PLAMO_API_KEY`を設定し、クライアントから`Authorization: Bearer ...`を付けてください。

## バックエンド

`--backend auto`（既定）は利用できるMetal GPUを選び、初期化できないときはCPUへ戻ります。`--backend metal`はGPUを必須とし、`--backend metal-fp32`も同じFP32経路を使います。`--backend cpu`は既存のCPUアセンブリ経路です。

```sh
python3 tools/server.py --backend metal --memory auto
```

翻訳本体はすべての経路で活性値・演算・累積がFP32です。元BF16重みのFP32への展開は正確で、FP16への丸めや追加の量子化はありません。加算順序が異なるためGPUのハッシュはCPU基準と一致しません。元BF16の共有メモリ保持、精度、検証範囲は[Metalバックエンド](METAL.md)を参照してください。

8BのMetal生成は、0.8Bのワーカーと重みがあれば、候補作成と8Bの一括検証を自動で使います。`plamo-download --model 0.8b --weights`で補助重みを用意できます。補助モデルだけはQ4で保持し、確定するトークンは元の8BがBF16重み・FP32演算で検証します。通常の逐次生成とはFP32の丸め差があります。`/health.speculative.enabled`・`block_tokens`で有効状態を確認でき、`PLAMO_SPECULATIVE=0`で無効化できます。補助モデルが利用できない場合や通信が切れた場合は通常生成へ戻ります。[設定・メモリ・数値検証](METAL.md#8bの候補検証による生成)を参照してください。

翻訳結果の`speculative`には`batches`・`consumed_tokens`・`proposed_tokens`・`accepted_tokens`・`draft_seconds`も返します。採用候補が複数あると1回の検証で複数トークンを確定するため、`timings.decode_steps`とコマンド数は一致しません。`decode_steps / decode_seconds`は補助処理、8B検証、KV反映を含む生成速度です。`compute.verification_batches`・`verified_tokens`・`committed_tokens`はGPU側の累積値です。

`/health`の`compute`にバックエンド・GPU名・精度モード・`gpu_dispatches`・`command_buffers`・`fused_mlps`・`fused_decodes`・`cpu_fallbacks`・`errors`・累積`gpu_seconds`を返します。`fused_decodes`は全層の生成を一つのGPUコマンドで実行したステップ数です。`weight_storage`は`bf16`または`lossless-bf16`、`packed_weight_bytes`は圧縮して保持する元データ量です。`disk_cache_hits`・`disk_cache_weight_bytes`・`disk_cache_storage_bytes`は[事前作成した重みファイル](WEIGHT_CACHE.md)からの累積読み込み回数・元BF16換算の量・保存形式での量です。`allocated_bytes`はMetalの確保量、`recommended_working_set_bytes`はデバイスの推奨量、`residency_sets`はGPUリソースの常駐管理が有効かを示します。数値は起動時・翻訳終了時の通知で更新します。GPUの初期化・実行エラーの理由は`compute.note`に表示します。

## CPUスレッド

この節は`--backend cpu`を選んだ場合の設定です。GPUのスレッド数は`--threads`では指定しません。

大きなBF16行列の出力行を複数スレッドに分け、既存のアセンブリカーネルを並列実行します。既定はmacOSが報告する性能コア数です。取得できない場合は最大4スレッドにします。`--threads 1..64`で明示でき、`--threads 1`では逐次実行します。スレッド作成に失敗した場合は作成できた数で動作し、実際の数を`/health`の`threads`に表示します。

```sh
python3 tools/server.py --model 2b --backend cpu --memory auto --threads 4
```

重みの取得・保持領域の調整は呼び出し元の1スレッドが担当します。取得済みの重みを全計算スレッドが使い終えてから次へ進みます。小さな行列ではスレッドを起こしません。各出力値のFMA・加算順序、Attentionの因果順序、greedy生成は維持します。通常のCLIも同じエンジンと設定を使います。従来の1スレッドのアセンブリCLIは`plamo-asm`として残しています。

重みの多くが都度読み込みになる場合はI/Oが律速し、コア数に比例して速くなるわけではありません。比較結果と再現手順は[最適化の記録](OPTIMIZATION.md#lm-studioとの速度差とcpu並列化)にあります。

## メモリモード

3つのモードはCPU・Metalの両方で使えます。以下はCPUの匿名メモリ・付け替え方式の説明です。Metalは元BF16を共有バッファへ直接読み込み、`resident`では2次元テンソルを先読みします。GPUの場合もOSの余裕とデバイスの作業領域に収まる必要があります。詳しくは[GPUのメモリ保持](METAL.md#共有メモリと重みの保持)を参照してください。

| モード | CPUでの動作 |
| --- | --- |
| `auto`（既定） | 空きメモリに応じた上限の領域へ、保持できる行列を起動時に先読みして再利用する。残りは小さな固定領域へ都度読み込む。実際のモード名は`adaptive` |
| `resident` | 元のBF16ファイルを匿名メモリへ読み込み、すべてのテンソルをその領域へ付け替える。余裕が不足する場合は起動失敗にする |
| `stream` | ファイルのmmap・FD・語彙は保持し、大きな射影は最大1 GiBの領域へまとめて読む従来方式 |

8Bの4ファイルは合計 **17,063,540,224バイト（約15.89 GiB）** です。`resident`の起動時判定は、これに2 GiBを加えた **19,211,023,872バイト（約17.89 GiB）** を使います。リクエストのKV・入力処理用メモリも必要です。

他モデルの常駐判定（重み＋2 GiB）は0.8Bで3.66 GiB、2Bで7.36 GiB、31Bで61.15 GiBです。KV・入力処理・OSのメモリは別途必要です。31Bの全重みは約63.51 GBあり、24 GiBのMacでは常駐できません。31Bの`stream`では1 GiBを超える行列をmmapから直接読みます。

判定にはmacOSのfreeページとfile-backedページを使い、重なる可能性のあるspeculativeページを差し引いて保守的に見積もります。数値はOSが再利用できる量の推定であり、確保の保証ではありません。

`auto`は起動時の空きから2 GiBを引いた量を絶対上限にします。保持用と都度読み込み用を同じ匿名領域に置き、そのうち使える範囲だけを有効にします。アドレス空間の予約だけでは全重みを物理メモリへ載せません。読み込み用の領域は起動時の余裕の約1/8、1〜64 MiBとし、残りに収まる重みを保持します。空きが減ってもこの読み込み領域は維持します。重みを取得するときは約250 ms間隔、待機中は約1秒間隔で空きを確認し、保持済みのページと現在の空きから有効範囲を再計算します。未使用の仮想領域は空きの計算に加えません。OSの圧縮・回収によって空きが増えても、起動時の絶対上限を超えて拡張しません。

空きが減れば保持データの末尾を解放し、空きが増えれば保持できる範囲を広げます。通常のミスでは既存の保持データを入れ替えません。モデル全体を毎トークン順に走査するため、単純なLRUでは全データが毎回追い出されるためです。APIは保持可能な行列を起動時に先読みし、保持されなかった重みは要求時に読み込み領域を再利用します。大きな行列もその領域に収まる行数へ分割し、各出力値の加算順序は維持します。小さな正規化重みやEmbeddingの参照などは元のmmapを使います。

`auto`も`resident`も8 MiB以下の`pread`と`F_NOCACHE`でファイルキャッシュとの二重保持を抑えます。BF16の精度・バイト列は変換しません。保持できた範囲は読み取り専用です。`resident`は全ファイルの先読みが成功してから元のmmap・FDを解放し、推論の明示的な重み読み込みは0になります。読み込み中に余裕が減った場合は新しい領域を解放して起動を中断します。

これらは`mlock`による物理メモリ固定ではありません。OSは圧縮・スワップでき、重み領域の上限はプロセス全体のRSS上限ではありません。KV・入力処理・語彙・元ファイルのページも使います。`auto`の保持量の縮小・回復には再起動は不要です。起動時の絶対上限を引き上げる場合や、`resident`・`stream`へのモード変更は再起動で行います。

## 初回応答を速くする起動時の先読み

APIの既定の`--memory auto`は、CPU・Metalの両方で、保持可能な行列を読み込んでからワーカーの`ready`を通知します。先読み中の`GET /health`は`loading`・503、翻訳要求も503で即座に応答します。起動が長くなる代わりに、初回要求の重み読み込みを減らします。`--startup-timeout`（既定180秒）は先読みを含みます。

QKV・Attention出力・MLP・出力ヘッドを推論で使う順に読み込み、推論時と同じ行ブロック境界で保持します。上限に収まらない範囲は読み飛ばすため、31Bなど全重みが収まらないモデルも部分的に先読みできます。巨大なEmbedding全体は先読みせず、入力で選ばれる行を元のmmapから読みます。Embeddingと出力ヘッドが共有されるモデルでは、ヘッドとして先読みします。

GPUでは保持ブロックを読み込むごとに各ページへの初回アクセスを完了させ、大量の未準備バッファを後回しにしません。ページ準備には専用の小さなカーネルを使います。その後、全行列を専用バッファまたは実行用ファイルからGPUで参照できる場合は、英日・日英の固定接頭12トークンを計算し、K・V・回転済みKを保存します。要求時はトークン列が完全一致する接頭だけを復元し、本文以降を計算します。本文や直前の翻訳結果をキャッシュする仕組みではありません。BF16の元バイト列とFP32演算を維持します。起動後も空きの減少に合わせて保持量を縮小するため、待機中のメモリ圧によっては初回要求でも一部の読み込みが発生します。

`--memory resident`は従来どおり完全先読みを要求します。`--memory stream`は起動時の重み先読みを行いません。単発CLIの`auto`は最初の入力処理中に保持可能な行列を準備します。固定接頭2方向の事前計算は常駐APIだけで行います。

ウォームアップなしの起動・初回・2回目を別々に測るには:

```sh
python3 tools/benchmark_startup.py --model 2b --backend metal --memory auto --report build/startup-2b.json
```

`/health`の`prompt_cache`には`en_ja_tokens`・`ja_en_tokens`・`bytes`・`prepare_seconds`を表示します。準備時間は`load_seconds`・`preload_seconds`にも含まれます。8Bの固定接頭は2方向で約4.7 MBです。必要なGPU射影が解決できない場合、CPU、`stream`では事前計算を省きます。`PLAMO_PREFIX_CACHE=0`で固定接頭の再利用を無効にし、同じバイナリで比較できます。単発CLIの起動込み待ち時間は`tools/benchmark_cli.py`で別に測定できます。

Apple M2・24 GiBで、`Hello.`を起動完了直後に送り、初回と2回目を測りました。ダミー要求・キャンセル・トークナイザー検査を先に実行せず、3プロセスずつの中央値です。初回トークンの時間は`ready`後の要求送信から計測し、起動時間は別に記録しています。

| モデル・バックエンド | 起動時間：前→後 | 初回トークン：前→後 |
| --- | ---: | ---: |
| 0.8B・Metal | 0.103→0.402秒 | 0.463→0.118秒 |
| 2B・Metal | 0.110→1.089秒 | 1.590→0.299秒 |
| 8B・Metal | 0.199→6.777秒 | 8.748→1.151秒 |
| 2B・CPU 4スレッド | 0.093→0.991秒 | 1.342→0.420秒 |

0.8B・2Bは計算用行列をすべて先読みでき、初回要求中の明示的な大行列読み込みは0でした。8Bは約15.78〜15.85 GBを保持し、初回要求の読み込みは変更前の約16.52〜18.08 GBから約0.67〜0.81 GBになりました。8Bでは測定時の空きと保持量も異なるため、数値は今回のメモリ条件での結果です。重み全体が収まらない場合の読み込みや、起動後のメモリ圧は残ります。

この変更は先読みとGPU準備を起動へ移すものです。起動時間も含めて用途に合うか判断してください。各回の時間・保持量・読み込み量・同一トークン列の検証は[起動時先読みの測定結果](benchmarks/startup.json)に記録しています。

さらに無損失キャッシュ、正規化の並列集計、生成全体のGPU化後は、ウォームアップなしの初回トークンが2Bで0.291→0.202秒、8Bで1.374→0.660秒でした。2Bの起動は1.156→1.132秒、8Bは6.640→8.445秒です。8Bの変更前後は起動時の空きと保持量が異なり、変更後は計算用16.18 GBを約12.30 GBのキャッシュに保持して大行列の読み込みを省けました。3回の中央値と各回の条件は[追加最適化の測定結果](benchmarks/decode.json)に記録しています。

## 翻訳

```sh
curl http://127.0.0.1:8000/translate \
  -H 'Content-Type: application/json' \
  -d '{"text":"Hello.","source_lang":"en","target_lang":"ja","max_tokens":512}'
```

`translation`、`finish_reason`、入力・生成トークン数、トークン化・入力処理・生成の時間、明示的な重みI/O、メモリモードと終了時のメモリ情報が返ります。`weight_io`の`cache_hits`・`cache_hit_bytes`・`cache_misses`はその要求中の値です。`finish_reason`が`length`なら生成上限で止まっており、翻訳が完了したことを意味しません。

`timings.decode_seconds`は生成の経過時間、`decode_steps`はモデルを追加実行した回数です。`decode_steps / decode_seconds`で起動・入力処理を除いた生成速度を計算できます。生成中の重みの再読み込み時間は含みます。`decode_gpu_seconds`・`decode_command_buffers`・`decode_gpu_dispatches`は生成中だけのGPU時間・同期回数・カーネル数、`weight_io.decode_staged_bytes`・`decode_read_calls`は生成中だけの明示的な重みI/Oです。起動準備や入力処理の累積値を含みません。`prefill_gpu_seconds`・`prefill_command_buffers`・`prefill_gpu_dispatches`は要求中の入力処理だけの値です。`usage.cached_prompt_tokens`は固定接頭から再利用した入力トークン数で、`prompt_tokens`には再利用分も含みます。

8Bの部分保持では、[事前作成した重みファイル](WEIGHT_CACHE.md)をGPUから直接参照し、共有バッファへのコピーを省きます。全射影がそろい、作業領域がGPUの推奨量に収まれば、ファイル参照分を含めて全層1コマンドで実行します。`compute.disk_cache_mapped_bytes`は参照した保存領域の累積量、`weight_io.decode_mapped_bytes`はその要求の生成中だけの量です。`compute.disk_cache_mapping_bytes`は再利用のため保持しているGPUファイルビューの範囲で、専用コピーの予算とは別です。GPUの確保量は`compute.allocated_bytes`にも反映されます。これらは物理ディスクI/Oではなく、`decode_staged_bytes`が0でもOSによるページインは発生し得ます。

| フィールド | 既定値・制約 |
| --- | --- |
| `text` | 必須。空文字・NUL・不正Unicodeを拒否。テンプレート込み4096 Unicode符号位置と、生成用プロンプト4096トークン以内 |
| `source_lang` / `target_lang` | `en` / `ja`。`ja` / `en`にも対応 |
| `max_tokens` | 512。1〜4096の整数。通常のCLIは既定256。`--max-tokens`で指定可能 |
| `stream` | false。trueならSSEで逐次出力 |

`POST /v1/translate`も同じ動作です。JSON本文の上限は64 KiBです。1ワーカーにつき推論は1件ずつで、処理中に来た別の要求は429と`Retry-After: 1`を返します。`GET /health`は推論中でも応答します。

## ストリーミング

```sh
curl -N http://127.0.0.1:8000/translate \
  -H 'Content-Type: application/json' \
  -d '{"text":"Hello.","stream":true,"max_tokens":512}'
```

`prefill`イベントで入力トークン数、`token`で出力片、`done`で最終結果と計測値を返します。BYTEトークンでUTF-8の途中が分割された場合も結合して返します。UTF-8として不正な生成バイトはJSONの表示上U+FFFDへ置換します。生成トークンとネイティブワーカーの元バイト列は変えません。

切断を検出するとキャンセルを送り、ワーカーは次の生成境界で停止します。進行中の入力処理や行列演算を途中で強制中断はしません。前の応答を回収してから次のリクエストを受け付けます。

推論前に入力の大きさを確認する場合:

```sh
curl http://127.0.0.1:8000/tokenize \
  -H 'Content-Type: application/json' \
  -d '{"text":"Hello.","source_lang":"en","target_lang":"ja"}'
```

## 検証と性能の範囲

0.8B・2Bでは実モデルの`resident`と`stream`で内部状態・logitsのハッシュが一致し、実HTTP APIの英日・日英翻訳を確認しています。31Bは全ヘッダー・ローダー・トークン化までで、実翻訳は未検証です。[モデル別の検証範囲](MODELS.md)を参照してください。

現在の`adaptive`も0.8B・2B・8Bで、キャンセル後の連続2要求の生成ID・内部状態・logitsを既存の`stream`の結果と照合しています。8Bの`Hello.`では、17.06 GBの重みファイルに対して起動時の上限を14.86 GBとし、14.76 GBを保持、64 MiBを読み込み領域として利用しました。21トークンの入力処理と1回の生成計算で、明示的な読み込みは従来の32.36 GBから2.84 GBになり、全4ハッシュは一致しました。これはその時の空きに応じた一部保持の検証例で、利用可能な保持量や速度は環境によって変わります。

アセンブリ本体の上にC・Objective-Cの常駐ワーカーとPython標準ライブラリのHTTP層を追加しています。HTTP要求ごとにCLIを起動する方式ではありません。CPUカーネルは手書きアセンブリ、GPUカーネルはMetalです。外部のMLライブラリは使いません。

- 元のBF16バイトのコピー、オフセットを保ったテンソルの付け替え、途中失敗時の全解放を合成ファイルで検証。
- `auto`の一部保持・繰り返し走査での再利用、空き減少時の解放と増加時の拡張、不完全な読み込みからの復旧を合成ファイルで検証。行分割した行列積・行列ベクトル積は分割前とビット単位で比較。
- 2/4スレッドの行列演算で符号付き入力・ストライド・端数・空形状・行ブロックを比較。重み取得が計算スレッドから呼ばれないことも検証。
- GPUの数値誤差・端数・ストライド・共有バッファの再利用・常駐管理・圧力による解放を合成データとMetal検証レイヤーで確認。実モデルのCPU/GPU差は[数値比較ツール](../tools/compare_backends.py)で測定。
- KV件数・RoPE状態のリセットと、重みやキャッシュ内容を不用意に消さないことを検証。
- `stream`の実モデルの同一プロセスへの2要求で、入力処理・生成後のlogitsと状態のハッシュが最適化前のアセンブリと一致。
- 別の長い入力を処理した後の`Hello.`も「こんにちは。」と正常終了することを確認。
- キャンセル後の次リクエストでも、同じ基準のハッシュへ戻ることを実モデルで確認。
- HTTP・SSE・分割UTF-8・同時要求・状態確認・キャンセル・ワーカー異常・入力拒否を、モデル不要のテストで検証。

以下の既存ハッシュとの比較と後述の測定は8Bのものです。0.8B・2Bも同じ検証ツールで連続要求を確認できます。

実モデルの再検証:

```sh
python3 tools/verify_api.py --model 2b
python3 tools/verify_api.py --model 2b --translate --memory auto
python3 tools/verify_api.py
python3 tools/verify_api.py --backend cpu --translate --memory auto --report build/api-8b-adaptive.json
python3 tools/verify_api.py --backend cpu --translate --memory stream
# 十分な空きがある環境で、先読み後の完全一致も検証する場合
python3 tools/verify_api.py --backend cpu --translate --memory resident
# スレッド数を変えた同じプローブを3回測定
python3 tools/verify_api.py --model 2b --backend cpu --translate --memory resident --threads 1 --repeats 3 --report build/api-serial.json
python3 tools/verify_api.py --model 2b --backend cpu --translate --memory resident --threads 4 --repeats 3 --report build/api-parallel.json
python3 tools/verify_api.py --backend metal --translate --memory auto --report build/api-metal.json
```

以前の全常駐か`stream`かを選ぶ実装での計測です。検証機は24 GiBですが、計測時に再利用可能メモリは約8.9 GBで、当時の`auto`は`stream`を選びました。585文字・テンプレート込み125トークンの入力は、入力処理約29.29秒、続く3回の生成計算約16.61秒でした。先頭4トークンまでの計測で、文章全体の翻訳時間ではありません。生成1ステップは約5.54秒でした。現在の`adaptive`の速度を示す測定ではありません。

全8B重みを匿名メモリへ先読みする実モデルの`resident`モードは、現在の検証機では余裕不足のため実行していません。そのモードの速度と実モデルのlogits一致は未検証です。合成ファイルによる元バイト保持・付け替え・演算結果の一致と、`stream`での実モデル検証を区別します。

Built with PLaMo. モデルの利用条件は[公式ライセンス](../LICENSES/PLaMo-Community.ja.txt)を参照してください。
