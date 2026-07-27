<div align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,12,20&height=180&section=header&text=RL%20Portfolio%20Trading&fontSize=42&fontColor=fff&animation=twinkling&fontAlignY=35&desc=PPO%20%E2%80%A2%20DDQ%20%E2%80%A2%20Branching%20DDQ%20%E2%80%A2%20Vietnamese%20Stocks&descAlignY=56&descSize=17" width="100%"/>
  <p>
    <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
    <img src="https://img.shields.io/badge/PyTorch-Agents-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch"/>
    <img src="https://img.shields.io/badge/Task-Portfolio%20Allocation-6A5ACD?style=for-the-badge" alt="Task"/>
  </p>
</div>

# RL Portfolio Trading

Dự án nghiên cứu phân bổ danh mục cổ phiếu Việt Nam bằng reinforcement learning. Hệ thống so sánh ba tác tử dùng LSTM, chạy backtest theo ngày và lưu đầy đủ artifact cho từng thí nghiệm.

> Chỉ phục vụ nghiên cứu và giáo dục. Không phải khuyến nghị đầu tư.

## Nội dung

- **PPO-LSTM**: policy liên tục sinh tỷ trọng mục tiêu cho từng cổ phiếu và tiền mặt.
- **DDQ-LSTM**: Double DQN với một hành động rời rạc tại mỗi bước.
- **Branching DDQ-LSTM**: Double DQN đa nhánh, quyết định mua/giữ/bán cho từng mã.
- Môi trường `Gymnasium` có phí giao dịch, giao dịch theo lô 100 cổ phiếu và các reward hướng rủi ro.
- Dữ liệu 16 mã: `ACB`, `BID`, `BVH`, `CTG`, `FPT`, `GAS`, `HPG`, `MBB`, `MSN`, `MWG`, `SHB`, `SSI`, `STB`, `VCB`, `VIC`, `VNM`.
- Chia dữ liệu theo thời gian: 70% train, 15% validation, 15% test.

## Quy ước backtest

Để tránh same-bar look-ahead bias, môi trường dùng chuỗi sự kiện sau:

1. Quan sát tại ngày `t` chỉ gồm dữ liệu đến hết ngày `t`.
2. Hành động được khớp tại giá mở cửa ngày `t + 1`.
3. Giá trị danh mục và reward được ghi nhận tại giá đóng cửa ngày `t + 1`.

PPO dùng action liên tục có kích thước `N + 1`, gồm `N` mã và một tỷ trọng tiền mặt. DDQ dùng action rời rạc `sell`, `hold`, `buy`; Branching DDQ chọn một action rời rạc cho mỗi mã.

## Cài đặt

Cần Python và `pip`.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Lệnh trên cài PyTorch CPU. Nếu dùng CUDA, cài bản `torch` phù hợp môi trường trước khi chạy `pip install -r requirements.txt`.

## Chuẩn bị dữ liệu

Repository chỉ lưu archive 7 đặc trưng tại `data/processed/dataset_trading.zip`. Các cấu hình train hiện tại dùng 9 đặc trưng trong `data/processed_v2`:

```text
close_norm, return_1d, return_5d, return_20d, macd,
rsi, adx, volume_norm, volatility_20d
```

Tạo dataset đầy đủ trước khi train:

```bash
unzip -o data/processed/dataset_trading.zip -d data/raw
python scripts/generate_extended_dataset.py
```

Script đọc `data/raw/*.csv`, chuẩn hóa dữ liệu, tính thêm `return_20d` và `volatility_20d`, rồi ghi 16 file vào `data/processed_v2`.

### Tải dữ liệu mới

`src/data/download_data.py` tải OHLCV từ `vnstock`. Có thể đặt `API_KEY_VNSTOCK` trong `.env` nếu nguồn dữ liệu yêu cầu đăng ký; khóa không được commit.

```env
API_KEY_VNSTOCK=your_key
```

Sau khi tải, xử lý dữ liệu thành 9 đặc trưng bằng `scripts/generate_extended_dataset.py`.

## Chạy huấn luyện

Từ thư mục gốc repository:

```bash
python -m src.training.PPO --config Conf/ppo_conf.yaml
python -m src.training.DDQ --config Conf/ddq_conf.yaml
python -m src.training.BranchingDDQ --config Conf/branching_ddq_conf.yaml
```

Cấu hình YAML kiểm soát tickers, features, dữ liệu, reward, tham số LSTM, schedule học và checkpoint. Không sửa config của một run đã hoàn thành khi muốn đánh giá lại; `config.json` trong thư mục run là nguồn cấu hình của run đó.

Các notebook tương ứng:

- `notebooks/01-train-ppo.ipynb`
- `notebooks/02-train-ddq.ipynb`
- `notebooks/03-train-branching-ddq.ipynb`
- `notebooks/04-compare-ddq-and-branching-vs-untrained.ipynb`
- `notebooks/data_analysis.ipynb`

## Đánh giá checkpoint DDQ

Hai tác tử DDQ hỗ trợ chế độ evaluate-only. Không truyền `--run-dir` hoặc `--checkpoint` để tự chọn run hợp lệ mới nhất trong `results/runs`.

```bash
python -m src.training.DDQ --eval --run-dir results/runs/<ddq_run>
python -m src.training.BranchingDDQ --eval --run-dir results/runs/<branching_ddq_run>
```

Có thể chỉ định checkpoint riêng:

```bash
python -m src.training.DDQ --eval --checkpoint path/to/best_model.pt
```

Branching DDQ còn lưu lịch sử tài khoản CSV vào thư mục `eval/` của run.

## Artifact của một run

Mỗi lần train tạo `results/runs/<run_id>/`:

```text
config.json          cấu hình đã dùng
training.log         log huấn luyện
checkpoints/         best_model.pt, final_model.pt, checkpoint_*.pt
metrics.csv          metrics theo episode
eval_metrics.csv     metrics validation và test
train_steps.csv      metrics mỗi bước cập nhật
summary.json         kết quả tổng hợp cuối run
```

`best_model.pt` là checkpoint có Sharpe validation tốt nhất khi checkpoint đó tồn tại. Final test ưu tiên checkpoint này; nếu không có, dùng trọng số cuối quá trình train.

## Cấu trúc repository

```text
Conf/                cấu hình thí nghiệm
Docs/                tài liệu bổ sung
data/                archive, dữ liệu raw và dữ liệu đã xử lý
notebooks/           notebook phân tích, train và so sánh
scripts/             tạo dataset và helper dashboard
saved_models/        model mẫu
src/agents/          PPO, DDQ và Branching DDQ agents
src/data/            tải và tiền xử lý dữ liệu
src/environment/     action, state, reward và TradingEnv
src/models/          kiến trúc LSTM
src/training/        entry point train và evaluation
tests/               regression test logic
```

## Kiểm tra

```bash
python -m unittest discover -s tests
```

## Reward và chỉ số

`TradingEnv` hỗ trợ `advanced`, `sharpe` và `sharpe_plus`. Các reward đều xét lợi nhuận so với benchmark equal-weight cộng tiền mặt; tùy loại có thêm phạt biến động, drawdown và turnover.

Kết quả evaluation gồm tổng lợi nhuận, lợi nhuận năm hóa, volatility, Sharpe, Sortino, Calmar, max drawdown, win rate, profit factor, VaR và CVaR. Báo cáo cũng so sánh tác tử với `equal_weight` và `buy_and_hold_equal_weight`.

---

<div align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=120&section=footer" width="100%"/>
  <em>Reinforcement learning experiments for portfolio allocation.</em>
</div>
