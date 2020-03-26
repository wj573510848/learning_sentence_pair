模型来自[大佬的文章](https://mp.weixin.qq.com/s?__biz=MjM5ODkzMzMwMQ==&mid=2650412629&idx=1&sn=d5e182941286af6adb745d8393f35151&chksm=becd900f89ba19199ac6c4fb31a2717d05363ebdbf5371f5dd5ec03d6af1e4ddd28c1dc1ad35&mpshare=1&scene=1&srcid=0323B3pt3BmrC9tFR9kUeXJ9&sharer_sharetime=1584929505368&sharer_shareid=0f471e468d6ec7f4d9808b9cb78b0843&exportkey=AdCkgGksVtM6hGSX2GF7jtc%3D&pass_ticket=zNlwxe4C4sxo0xTnJTB6g7OMlHBKaSYcEl4rdM9nkzWZdih384RUZNazBIMfZ7nR#rd)，[论文](https://arxiv.org/abs/2002.02925)使用pytorch实现的，大佬使用[tf]((https://github.com/qiufengyuyi/bert-of-theseus-tf))实现了，在这基础之上，我探索了一下finetune的细节。

# 环境

ubuntu 16.04

python 3.6

tensorflow 1.15

# bert_of_theseus代码修改细节

大佬在[知乎专栏](https://zhuanlan.zhihu.com/p/112787764)讲解了模型的修改之处。

主要有两点：

1. model replace training
    
    * `modeling_theseus.transformer_model_theseus`使用s-model 替换 p-model，使用的是随机采样（或论文中的计算公式）。

    * `modeling_theseus.get_assignment_map_from_checkpoint_for_theseus`,s-model的初始化


2. freeze p-model参数，只更新 s-model参数（`optimization_theseus.create_optimizer_for_bert_theseus`）

    第1阶段finetune更新全部参数，第二阶段finetune，只更新s-model的参数

# LCQMC训练&测试

1. 压缩到6层 

    ```bash
    python run_classifier.py --do_train --do_eval suc_layers = 6
    python run_classifier.py --do_train --do_eval suc_layers = 6 --finetune_suc
    ```
    
    stage1: lr:2e-5 train_batch_size:64 epoch:10 eval_accuracy:0.8612

    stage2: lr:2e-5 train_batshc_size:64 epoch:4 eval_accuracy:0.86 并没有提升 业务数据：0.53435 过拟合了？


    stage1: lr:2e-5 train_batch_size:64 num_train_epochs:4 eval_accuracy:0.83

    stage1: lr:2e-5 train_batch_size:64 num_train_epochs:3 eval_accuracy:0.85536

2.  压缩到两层

    ```bash
    python run_classifier.py --do_train --do_eval --stage_1_outputdir out_1_2 --stage_2_outputdir out_2_2 --suc_layers 2 
    python run_classifier.py --do_train --do_eval --stage_1_outputdir out_1_2 --stage_2_outputdir out_2_2 --suc_layers 2 --finetune_suc
    ```
    
    stage1: lr:2e-5 train_batch_size:64 epoch:10 eval_accuracy:0.78584

    stage2: lr:2e-5 train_batshc_size:64 epoch:2 eval_accuracy:0.82528

    stage2: lr:2e-5 train_batshc_size:64 epoch:1 eval_accuracy:0.81032  

    stage2: lr:2e-5 train_batshc_size:64 epoch:3 eval_accuracy:0.82424  业务数据：0.7595

3. 其它 

    python run_classifier.py --do_train --do_eval --stage_1_outputdir out_1_3 --stage_2_outputdir out_2_3 --suc_layers 2

    stage1: lr:2e-5 train_batch_size:64 epoch:4 eval_accuracy:0.76512

# inference速度

1. left sentence 长度：60, right sentence 长度：60, batch_size:10

    bert 耗时: 1.2s

    suc_layers_2： 0.57s

2. left sentence 长度：60, right sentence 长度：60, batch_size:1

    bert 耗时:   0.18 
    
    suc_layers_2: 0.047