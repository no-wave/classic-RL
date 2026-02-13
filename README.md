# 강화 학습 1: Classic Reinforcement Learning
#### 강화학습 이론부터 MDP, MC, TD, Q-Learning, A2C 까지 핵심 가이드

<img src="https://beat-by-wire.gitbook.io/beat-by-wire/~gitbook/image?url=https%3A%2F%2F3055094660-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FYzxz4QeW9UTrhrpWwKiQ%252Fuploads%252FuCZSbmtsV76TFuH7Be0r%252FRL1.png%3Falt%3Dmedia%26token%3D7582913e-a36c-42ba-b121-ee5f5c1b8773&width=300&dpr=3&quality=100&sign=7e85da32&sv=2" width="500" height="707"/>


## 책 소개

강화학습(Reinforcement Learning)은 에이전트가 환경과 상호작용하며 시행착오를 통해 최적의 행동 전략을 스스로 학습하는 머신러닝의 한 분야다. 지도 학습이 정답 레이블로부터 패턴을 학습하고, 비지도 학습이 데이터의 내재적 구조를 발견한다면, 강화학습은 직접 경험하며 행동 방법을 배운다. 이는 인간과 동물이 세상을 배우는 방식에 가장 가까운 학습 패러디그램이며, 진정한 자율성과 적응력을 가진 AI로 가는 핵심 경로다.

강화학습은 2010년대 중반 DeepMind의 AlphaGo가 세계 바둑 챔피언을 이기는 순간 세상의 주목을 받았다. AlphaGo는 바둑의 규칙만 주어진 상태에서, 수백만 번의 게임을 통해 스스로 최적의 전략을 학습했다. 2017년의 AlphaZero는 이보다 한 단계 더 나아 체스, 바둑, 장기 세 종목 모두에서 기존 챔피언을 능가하는 성과를 달성했다. OpenAI의 Dota 2 AI는 팀 단위로 복잡한 전략 게임을 학습하며, 로봇 연구소에서는 강화학습으로 보행 로봇이 스스로 걷는 법을 배우는 모습이 공개되었다. 이러한 성과들은 강화학습이 단순한 학술적 연구가 아니라, 실제 세계의 복잡한 문제를 해결하는 강력한 도구임을 증명했다.

하지만 강화학습의 원리와 알고리즘을 체계적으로 이해하고, 실제로 구현할 수 있는 역량을 갖추는 것은 쉽지 않다. 마르코프 결정 과정, 벨만 방정식, 동적 프로그래밍, 몬테카를로 방법, 시간차 학습, Q-Learning, 정책 경사, Actor-Critic - 이러한 개념들은 강력하지만 추상적이고, 수학적 표기법으로 가득차 있다. 많은 독자들이 강화학습에 관심을 가지지만, 이론의 깊이와 실습 사이의 간격에서 좌절을 경험한다. 이 책은 그 간격을 메우는 것을 목표로 두었다.

본 서는 강화학습의 기초 이론부터 현대적인 딥러닝 기반 방법까지, 이론과 실습을 유기적으로 연결하여 체계적으로 다룬다. 각 장은 개념의 직관적 설명을 먼저 제공하고, 수학적 형식화를 거쳐, 실제로 동작하는 Python 코드로 구현하는 구조를 따른다. 단순히 알고리즘을 이해하는 것을 넘어, 독자가 스스로 강화학습 시스템을 구축할 수 있는 역량을 길러주는 것이 본 서의 궁극적 목표다. 모든 실습 코드는 Gymnasium(OpenAI Gym의 후속 프로젝트)과 PyTorch를 기반으로 작성되었으며, 독자가 직접 실행하고 실험할 수 있도록 설계했다.

이 책을 집필하면서 가장 중점을 둔 부분은 '직관'이다. 강화학습의 알고리즘들은 복잡해 보이는 수식으로 표현되지만, 그 뒤에는 깔끔한 직관적 논리가 있다. 예를 들어 벨만 방정식은 "현재 상태의 가치는 지금 받는 보상과 앞으로 받을 보상의 합"이라는 단순한 아이디어다. Q-Learning은 "최선의 행동을 가정하고, 그 행동이 가져오는 보상을 기반으로 학습한다"는 탐욕적 전략이다. REINFORCE는 "좋은 결과를 가져온 행동은 더 자주 해라"는 직관적 원칙이다. 각 장에서 수식을 제시하기 전에 이러한 직관을 충분히 설명한다. 독자가 "왜 이 수식이 이렇게 생겼는가?"라는 질문에 스스로 답할 수 있도록 하는 것이 목표다.

강화학습은 빠르게 발전하는 분야이다. 새로운 알고리즘, 새로운 환경, 새로운 응용이 계속 등장하고 있다. 본 서에서 다루는 클래식 강화학습의 범위는 유한하지만, 이 클래식 원리 위에 현대의 모든 발전이 세워져 있다. PPO, SAC, Model-based RL, Multi-agent RL 같은 고급 주제들은 본 서의 기초 위에서 자연스럽게 확장될 수 있다. 독자가 본 서를 마치고, 새로운 논문을 읽고, 새로운 알고리즘을 구현하는 역량을 갖추는 것이 진정한 성공이다.

마지막으로, 현재 우리는 LLM 기반 에이전트 시대의 문을 열고 있다. ChatGPT, GPT-4, Claude 같은 대규모 언어 모델은 자연어로 대화하며, 코드를 작성하고, 복잡한 작업을 자율적으로 수행하는 에이전트 시스템의 핵심이 되었다. 하지만 LLM 단독으로는 장기적 목표를 달성하기 위한 순차적 의사결정, 환경과의 상호작용을 통한 피드백 기반 학습, 특정 작업에 대한 최적 전략 학습이 불가능하다. 강화학습은 바로 이러한 빈자리를 채운다. RLHF(Reinforcement Learning from Human Feedback)로 LLM이 인간의 가치에 정렬되고, 강화학습 기반 agent가 복잡한 도구를 활용하며 장기적 목표를 달성하는 시대가 바로 현재다. 즉, 에이전트 시대에 강화학습은 선택적 기술이 아니라 핵심 기술이다. 본 서는 그 핵심 기술의 기초를 단단하게 쌓는 출발점이다.

AI Master 시리지의 Reinforcement Learning은 총 3편으로 기획되었으며 이책은 첫 편으로 클래식 강화학습 알고리즘을 서술한다. 나머지 2편은 Deep Reinforcement 이후의 강화학습, 3편은 LLM 에이전트에서 강화학습을 출판 예정이다.



## 목 차

저자 소개

Table of Contents (목차)

서문: 들어가며

Chapter 01: 강화 학습 개요

Chapter 02: 마르코프 결정 과정 (Markov Decision Process)

Chapter 03: 동적 프로그래밍 (Dynamic Programming)

Chapter 04: 몬테카를로 방법 (Monte Carlo Methods)

Chapter 05: 온-폴리시 몬테카를로 제어 (On-policy MC Control)

Chapter 06: 오프-폴리시 몬테카를로 제어 (Off-policy MC Control)

Chapter 07: 시간차 학습 (Temporal Difference Learning)

Chapter 08: Q-Learning

Chapter 09: n-스텝 부트스트래핑 (n-step Bootstrapping)

Chapter 10: 연속 상태 공간 (Continuous State Spaces) 문제

Chapter 11: Deep Q-Learning

Chapter 12: REINFORCE

Chapter 13: A2C (Advantage Actor-Critic)

Chapter 14: 클래식 강화 학습 요약

결론: 마무리 하며

References: 참고 문헌


## E-Book 구매

- Yes24: https://www.yes24.com/product/goods/178036563
- 교보문고: https://ebook-product.kyobobook.co.kr/dig/epd/ebook/E000012577109
- 알라딘: http://aladin.kr/p/RSQsp

## Github 코드: 

https://github.com/no-wave/classic-RL



