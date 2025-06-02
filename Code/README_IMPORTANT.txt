Hello.

In the link below, you will find all the code for all the attacks used for the benchmark results.

https://drive.google.com/drive/folders/1lylRKtTuZLoK2LZ3s6aEKSDaN4ImekRB?usp=sharing

Generally speaking, adversarial attacks can request either logits or softmaxes.
This is why you will have FGSM for softmaxes, and ElasticNet for logits.

The other attacks use exactly the same code; only the import, the attack instance, and the output directory name change.

For SoftMax, here is the list of attacks:
- FGSM (by default)
- BIM
- PGD
- AUTOPGD
- DeepFool
- NewtonFool
- JSMA
- Wasserstein

For Logits, here is the list of attacks:
- ElasticNet (by default)
- CarliniL0
- CarliniL2
- CarliniLinf
- Shadow Attack

Finally, the UAP, TUAP, and Adversarial Patch attacks have their own code, as their intrinsic logic differs from the rest of the attacks.

The dynamic_code folder contains code that allows you to test most attacks by simply changing the text parameter at the beginning of the code.
However, this code has not yet been tested and may require adjustments.
However, you will find instances of each attack inside, which you can copy and paste directly to repeat the tests 100% identically.