# AgrAmplifier

## Cite AgrAmplifier
If you use AgrAmplifier for research, please cite the following paper:

@article{shen2022better,<br />
  title={Better Together: Attaining the Triad of Byzantine-robust Federated Learning via Local Update Amplification},<br />
  author={Shen, Liyue and Zhang, Yanjun and Wang, Jingwei and Bai, Guangdong},<br />
  booktitle={Annual Computer Security Applications Conference (ACSAC ’22)},<br />
  year={2022},<br />
  publisher={ACM}<br />
}

## Code structure
To run the experiments, please run *dataset*_script.py, replace *dataset* with the dataset to experiment with.
The code can execute directly without downloading dataset file. List of runnable script files
* location_script.py (Refer to  Sec 4.1 LOCATION30, line 537 of the original paper)
  * Approximately need 3000 sec for each experiment
* cifar_script.py (Refer to Sec 4.1 CIFAR-10, line 529 of the original paper)
  * Approximately need 5700 sec for each experiment
* mnist_script.py (Refer to Sec 4.1 MNIST, line 534 of the original paper)
  * Approximately need 8000 sec for each experiment
* purchase_script.py (Refer to Sec 4.1 PURCHASE100, line 543 of the original paper)
  * Approximately need 18000 sec for each experiment
* texas_script.py (Refer to Sec 4.1 TEXAS100, line 550 of the original paper)
  * Approximately need 24000 sec for each experiment

__Note: the script is not optimized for CUDA, so recommended to run with CPU device. CPU is assumed to be an AMD Rayzen 4800H or higher, RAM working with DDR4-3200 16GB. With a higher spec device, the time consumption can be significantly reduced. On some LINUX device, attempts to run TEXAS100 and PURCHASE may fail due to limited RAM, recommended to run with 64GB or higher.__

E.g. in a LINUX environment, to execute the *LOCATION30*
experiment, please input the following command under the source code path

*python location_script.py*

Other files in the repository
* __constants.py__ Experiment constants, contains the default hyperparameter set for each expeirment
* __Defender.py__ Byzantine-robust aggregators, including
  * FL-Trust [1] (Refer to Sec 4.3.3, line 654 of the original paper, also in APPENDIX Algorithm 4)
  * Fang [2] (Refer to Sec 4.3.2, line 626 of the original paper, also in APPENDIX Algorithm 3)
  * Trimmed-mean [3] (Refer to Sec 2.2, line 236 of the original paper)
  * CosDen, EuDen, and MgDen (Refer to Sec 4.3.1, line 610 of the original paper, also in APPENDIX Algorithm 2)
* __FL_Models.py__ The models used for experiment, attacks are wrapped in this file, including
  * S&H attack [4] (Refer to Sec 4.2.1, line 581 of the original paper)
  * Label flipping (Refer to Sec 4.2.1, line 565 of the original paper)
  * Gradient ascent (Refer to Sec 4.2.1, line 570 of the original paper)
  * Mislead, the combined attack of label flipping and gradient ascent (Referenced as <L-flip>+<G-asc> in Sec 4.2.1 of the original paper)
  * Targeted attack [1] (Referenced as T-scal in Sec 4.2.2 of the original paper)

## Understanding the output
The output consists of for columns

| epoch | test_acc | test_loss | training_acc | trainig_loss |
|-------|----------|-----------|--------------|--------------|
| 0     | 0.08     | 4.1       | 0.12         | 3.14         |
| 10    | 0.42     | 1.3       | 0.39         | 1.52         |
| ...   | ...      | ...       | ...          | ...          |  
---
Namely, each column represents the corresponding value. A stand-alone experiment may not explain the effectiveness of the AGR.
To acquire experiment data shown in Table 1 (line 768 - 798), there are several steps you may need to follow
1. Select the dataset to experiment with, and open the corresponding experiment script. e.g. experiment with PURCHASE100 on __purchase_script.py__
2. Update the editable block of the script according to desired hyperparameters. You may leave it as is to reproduce Table 1 in the original paper
3. Choose attack mode, e.g. Mislead (L+G in Table 1, line 790)
4. Identify the AGR need to be addressed, e.g. FLTrust and FLTrust_Amp+ (column T & T+ in Table 1)
5. Update the experimented attack and experimented AGR in editable block of the script according to the selected attack and defense (with baseline included, see below example). The values shall come from __constants.py__, comments provided in __constants.py__.
   * att_experimented = []
   * agr_experimented = []

E.g. if you need to reproduce column T and T+ with <L+G> attack, line 790 of Table 1, you first go to __purchase_script.py__, update editable block into
```
# Number of iterations (communication rounds)
num_iter = 201
# Number of malicious participants
Ph = 50
# Size of hidden layer
hidden = 1024
# Fraction of malicious members
malicious_factor = 0.3
# Experimented attack
att_experimented = ["mislead"]
# Experimented AGR, details in constants.py
agr_experimented = [constants.baseline, constants.fang, constants.p_fang]
```
Then run the script.
Three csv files will be generated in directory __./output__ once script executed. One file with containing 'start_1000' in its filename is the baseline (note as file1). (There are overall 200 rounds but constants.baseline tell the script to start attack on round 1000). The rest two are those attacked and protected by corresponding AGR.
The file with 'def_fl_trust' in its filename (note as file2) corresponding to the AGR FLTrust [1] (Sec 4.3.3, line 654) while that with 'def_p-trust' corresponding to the AGR FLTrust_Amp+ (Sec 4.3.3, line 660).

To acquire the final value, compute the evaluation metric according to Sec 4.4.1, equation (1). The attack starts at round 50 so the T0 = 50 and T1 = 200, and we record every 10 rounds. Make difference of the attacked test accuracy (file2) and the baseline test accuracy (file1) after round 50 (the data is only meaningful after attack happens).
Now there are 16 differences we have from round 50 to round 200, sum the difference up and divide by 16. This will return the averaged test accuracy difference (\mathcal{L}) caused by attack under the protection of the AGR.


## Requirements
Recommended to run with conda virtual environment
* Python 3.7 / 3.8 / 3.9
* PyTorch 1.7.1
* numpy 1.19.2
* pandas 1.4.3

## Reference
[1] Xiaoyu Cao, Minghong Fang, Jia Liu, and Neil Zhenqiang Gong. 2020. FLTrust: Byzantine-robust Federated Learning via Trust Bootstrapping. arXiv preprint arXiv:2012.13995 (2020).

[2] Minghong Fang, Xiaoyu Cao, Jinyuan Jia, and Neil Gong. 2020. Local model poisoning attacks to byzantine-robust federated learning. In 29th {USENIX} Security Symposium ( {USENIX } Security 20). 1605–1622.

[3] Dong Yin, Yudong Chen, Ramchandran Kannan, and Peter Bartlett. 2018. Byzantine-robust distributed learning: Towards optimal statistical rates. In Inter- 1246 national Conference on Machine Learning. PMLR, 5650–5659.

[4] Virat Shejwalkar and Amir Houmansadr. 2021. Manipulating the Byzantine: Optimizing Model Poisoning Attacks and Defenses for Federated Learning. Internet 1223 Society (2021), 18.
