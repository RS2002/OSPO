# OSPO: One Step Policy Optimization

**Article:** "One Step is Enough: Multi-Agent Reinforcement Learning based on One-Step Policy Optimization for Order Dispatch on Ride-Sharing Platforms" (under review)



# 1. Workflow

![](./img/main.png)



## 2. Dataset

The dataset used in this study is derived from the [yellow taxi data in Manhattan](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page). The processed data can be found in the `./data` directory.



## 3. How to Run

```shell
python train.py
```

You can also set different parameters in the `process` function in `Worker.py` of `GRPO` to replicate the ablation study presented in our paper.



## 4. Parameters

The model parameters and training log files are located in the `./GRPO/parameters` and `./OSPO/parameters` directory.



## 5. Citation

```

```

