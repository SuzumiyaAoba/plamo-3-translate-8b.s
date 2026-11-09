# 常駐翻訳API

モデルと語彙を一度読み込んだワーカーを維持し、複数リクエストで再利用します。重み・モデル構造・greedy生成は変更していません。リクエスト間ではKVキャッシュの件数とRoPEの状態をリセットし、前の文章を次へ持ち越しません。

## 起動

公式モデルを取得済みなら、次で起動できます。

```sh
nix run github:SuzumiyaAoba/plamo-3-translate.s#serve -- --model 2b --memory auto --port 8000
```

初回の取得は `nix run github:SuzumiyaAoba/plamo-3-translate.s#download-model -- --model 2b --weights` です。`--model`には`0.8b`・`2b`・`8b`・`31b`を指定できます。省略時は`PLAMO_MODEL`または8Bです。1サーバーにつき1モデルを保持し、変更時は再起動します。データ保存先はCLIと同じ `PLAMO_DATA_DIR` を使います。開発時の `./models/plamo` を利用する場合:

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

`status`、`busy`、`model`、公式リポジトリ・固定リビジョン、`worker_pid`、読み込み時間、実際に選択したメモリモード、先に読み込んだ重みのバイト数、空きの推定値、フォールバックの理由が返ります。起動中でもHTTPの状態確認は応答します。

既定の待ち受けは127.0.0.1です。別のホストへbindする場合は`PLAMO_API_KEY`を設定し、クライアントから`Authorization: Bearer ...`を付けてください。

## メモリモード

| モード | 動作 |
| --- | --- |
| `auto`（既定） | 重みと2 GiBの余裕分が再利用可能メモリの推定値に収まる場合に先読みし、不足・読み込み失敗時は`stream`へ戻る |
| `resident` | 元のBF16ファイルを匿名メモリへ読み込み、すべてのテンソルをその領域へ付け替える。余裕が不足する場合は起動失敗にする |
| `stream` | ファイルのmmap・FD・語彙は保持し、大きな射影は最大1 GiBの領域へまとめて読む従来方式 |

8Bの4ファイルは合計 **17,063,540,224バイト（約15.89 GiB）** です。`resident`の起動時判定は、これに2 GiBを加えた **19,211,023,872バイト（約17.89 GiB）** を使います。リクエストのKV・入力処理用メモリも必要です。

他モデルの常駐判定（重み＋2 GiB）は0.8Bで3.66 GiB、2Bで7.36 GiB、31Bで61.15 GiBです。KV・入力処理・OSのメモリは別途必要です。31Bの全重みは約63.51 GBあり、24 GiBのMacでは常駐できません。31Bの`stream`では1 GiBを超える行列をmmapから直接読みます。

判定にはmacOSのfreeページとfile-backedページを使い、重なる可能性のあるspeculativeページを差し引いて保守的に見積もります。数値はOSが再利用できる量の推定であり、確保の保証ではありません。読み込み中も余裕を確認し、圧迫が増えた場合は新しい領域を解放して中断します。

先読みは8 MiB単位の`pread`と`F_NOCACHE`で行い、ファイルキャッシュとの二重保持を抑えます。読み込んだ全ファイルは読み取り専用にし、元ファイルのmmap・FDを解放します。重みの精度・バイト列は変換しません。`resident`では推論の明示的な重み読み込みは0になります。

`resident`は`mlock`による物理メモリ固定ではありません。他のアプリなどで圧迫が増えるとOSは圧縮・スワップできます。重みを置く余裕がない環境では、API化しても生成速度は大きく変わりません。

モードの選択は起動時です。ほかのアプリを終了するなどして余裕を増やした後は、サーバーを再起動してください。稼働中の自動昇格や、要求ごとの重み再ロードは行いません。

## 翻訳

```sh
curl http://127.0.0.1:8000/translate \
  -H 'Content-Type: application/json' \
  -d '{"text":"Hello.","source_lang":"en","target_lang":"ja","max_tokens":512}'
```

`translation`、`finish_reason`、入力・生成トークン数、トークン化・入力処理・生成の時間、明示的な重みI/O、メモリモードが返ります。`finish_reason`が`length`なら生成上限で止まっており、翻訳が完了したことを意味しません。

| フィールド | 既定値・制約 |
| --- | --- |
| `text` | 必須。空文字・NUL・不正Unicodeを拒否。テンプレート込み4096 Unicode符号位置と、生成用プロンプト4096トークン以内 |
| `source_lang` / `target_lang` | `en` / `ja`。`ja` / `en`にも対応 |
| `max_tokens` | 512。1〜4096の整数。元の単発CLIの上限256とは別 |
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

アセンブリ本体の上にCの常駐ワーカーとPython標準ライブラリのHTTP層を追加しています。HTTP要求ごとにCLIを起動する方式ではありません。推論カーネルは既存の手書きアセンブリです。外部のMLライブラリは使いません。

- 元のBF16バイトのコピー、オフセットを保ったテンソルの付け替え、途中失敗時の全解放を合成ファイルで検証。
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
python3 tools/verify_api.py --translate --memory stream
# 十分な空きがある環境で、先読み後の完全一致も検証する場合
python3 tools/verify_api.py --translate --memory resident
```

検証機は24 GiBですが、計測時に再利用可能メモリは約8.9 GBで、`auto`は`stream`を選びました。585文字・テンプレート込み125トークンの入力は、入力処理約29.29秒、続く3回の生成計算約16.61秒でした。先頭4トークンまでの計測で、文章全体の翻訳時間ではありません。生成1ステップは約5.54秒で、単にサーバーを常駐させても、このI/O中心の時間は残りました。

全8B重みを匿名メモリへ先読みする実モデルの`resident`モードは、現在の検証機では余裕不足のため実行していません。そのモードの速度と実モデルのlogits一致は未検証です。合成ファイルによる元バイト保持・付け替え・演算結果の一致と、`stream`での実モデル検証を区別します。

Built with PLaMo. モデルの利用条件は[公式ライセンス](../LICENSES/PLaMo-Community.ja.txt)を参照してください。
