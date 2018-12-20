# Creative-Adversarial-Networks
포항공대 정보통신연구소에서 인턴 프로젝트 (AI Art Lab 그림 모델)

## [그림 넣기]

## dataset
dataset 으로 [WIKIART](https://www.wikiart.org/) 를 사용 [Download](https://github.com/cs-chan/ArtGAN/tree/f5d6f6b58a6d8a4bd05aaaedd9688d08c02df8f2/WikiArt%20Dataset)

CLI 명령어
```shell
mkdir data
cd data
wget http://www.cs-chan.com/source/ICIP2017/wikiart.zip
unzip wikiart.zip
```


## Turing test
- 총 8개의 그림 중 4개의 그림은 사람이 4개의 그림은 저희 모델인 CAN 이 생성한 그림입니다. 

![turingTest](https://user-images.githubusercontent.com/45627868/50278002-3fdd2900-0489-11e9-8239-6018ae707052.PNG)




## reference 
[CAN: Creative Adversarial Networks, Generating "Art" by Learning About Styles and Deviating from Style Norms](https://arxiv.org/abs/1706.07068)
