## Entropy

### Motivation

가격은 수요와 공급의 힘에 대한 정보를 전달한다. 완전 시장이라면 가격을 예측할 수 없다. 
각 관측값이 가격 또는 서비스에 대해 알려진 모든 것을 전달하기 떄문이다. 
시장이 완전하지 못하면 가격은 부분적 정보로 형성되고, 특정 에이전트가 남들보다 더 많은 정보를 가지므로 그 정보의 비대칭성을 이용할 수 있다. 
가격 시계열의 정보 내용을 추정하고, 발생 가능한 결과를 Machine Learning Algorithm이 학습하는 데 기초가 되는 특성들을 만드는 것은 도움이 될 것이다. 
예를 들어서, Machine  Learning 알고리즘은 가격에 담긴 정보가 거의 없을 경우 모멘텀 베팅이 더 수익을 내리라는 것을 발견하고, 가격에 충분한 정보가 담긴 경우에는 평균 회귀 베팅이 더욱 수익을 내리라는 것을 알 수 있다. 
이번 장에서는 가격 시계열에 담긴 정보의 양을 찾아내는 방법에 대해 살펴보도록 한다.

### 시간 척도로서의 엔트로피

금융시장의 단일 증권의 시간으로서의 엔트로피를 생각해 보자. 이 개념은 물리학에서의 시간에 대한 정의로부터 출발한다.
물리학에서 시간은 이벤트와 이벤트 사이의 간격으로 정의된다. 

$$\Delta t = t(E_2) - t(E_1)$$

입자의 초기 상태는 $|\psi_i\rangle$로 표시되며, 어떤 사건 후의 상태는 $|\psi_f\rangle$로 표시된다. 상태 변화는 시간 진화 연산자 $U(t, t_0)$에 의해 설명되며, 여기서 $t_0$은 초기 시간이고 $t$는 최종 시간이다.

$$|\psi_f\rangle = U(t, t_0) |\psi_i\rangle$$

시간 진화 연산자는 다음과 같이 주어진다.

$$U(t, t_0) = e^{-\frac{i}{\hbar} H (t-t_0)}$$

여기서, $\hbar$는 축소 플랑크 상수, $i$는 허수, $H$는 시스템의 해밀토니언이다. 
시간을 만약 측정이벤트로 정의할 경우, 측정 연산자 $\hat{A}$와 그 고유 상태 $|a\rangle$를 가정한다. 
$\hat{A}$의 측정 결과가 $a$ 값으로 나타날 경우, 초기 상태 $|\psi_i\rangle$를 기준으로 이 결과의 확률은 다음과 같이 나타낼 수 있다.

$$P(a) = |\langle a | \psi_i \rangle|^2$$

결과가 $a$일 경우, 시스템의 상태는 고유 상태 $|a\rangle$로 붕괴된다. 열역학 제 2법칙에 의하면, 엔트로피는 시간이 지남에 따라 지속적으로 증가 혹은 최소한 감소하지 않는다는 것이 증명되었다.
$\Delta S$를 엔트로피의 변화량이라고 할 때, 다음과 같이 열역학 제 2법칙을 정의할 수 있다.

$$\Delta S \geq 0$$

이는 엔트로피가 시간이 흐른다는 강력한 증거 중 하나로 작용한다.

### 증권가격의 '시간'

한국 유가증권시장을 생각해 보자. 거래소는 일반적인 영업일 기준으로 오전 8시 30분부터 9시 사이에 동시호가 시장이 개장되면 오전 9시에 정규장 개장과 함께 동시호가 시장에서의 주문이 단번에 체결된다.
정규장은 오전 9시부터 오후 3시 20분까지 진행되며, 오후 3시 20분부터 3시 30분 사이에는 장마감 동시호가가 진행되며, 애프터마켓은 오후 4시까지 진행된다.
장외시장까지 전부 계산하면 오후 6시까지 모든 거래가 완료되며, 대부분의 거래는 정규장 개장 직후 30분과 정규장 마감 30분 전에 빈번하게 발생한다. 
여기서 주목할 점은, 하루는 24시간인데 증권이 거래되는 시간은 하루 중 10시간이 채 되지 않는다는 점이다. 가격의 변화를 동일한 시간 간격으로 관측한다는 것은 결국 그 자체가 왜곡이 될 가능성이 크다는 것을 알고 있어야 한다. 
가장 좋은 최소한의 이벤트 단위는 1 거래이다. 거래는 개개인의 경제주체가 독립적인 판단에 의해 결정되는 것이며, 의사결정의 주체가 선택 편향이 있다고 해도 최종 판단은 결국 독립적으로 발생하기 때문이다.

가격을 하나의 입자라고 생각하면, 다음과 같이 식 유도가 가능하다. 시장의 초기상태를 $|\psi_i\rangle$로 가정하자. 금융시장의 시간은 현실 시간과 괴리가 있기 때문에 상태 변화는 거래의 변화로 proxy를 생성한다.

$$|\psi_f\rangle = U(v, v_0) |\psi_i \rangle$$

여기서 $|\psi_i\rangle$는 초기 시장 상태, $|\psi_f\rangle$는 최종 시장 상태이다. 시간 진화 연산자는 거래량에 따른 시장 변동성 $V$를 사용하여 다음과 같이 표현할 수 있다.

$$U(v, v_0) = e^{-\frac{i}{\hbar} V (v-v_0)}$$

이때 $\hbar$는 금융 시장에서의 reduced plank constant에 대응되는 조정 상수로 간주된다. 측정 이벤트는 특정 가격 $p$에서의 거래를 의미하며, 거래 연산자 $\hat{P}$와 그 고유 상태 $|p\rangle$를 가정한다. 거래 결과가 $p$로 나타날 경우, 초기 상태 $|\psi_i\rangle$를 기준으로 이 결과의 확률은 다음과 같이 계산된다.

$$P(p) = |\langle p | \psi_i \rangle|^2$$

결과가 $p$일 경우, 시장 상태는 해당 가격의 고유 상태로 붕괴된다.

$$\text{Post-transaction state: } |p\rangle$$

여기서 $\Delta v \neq \Delta t$임으로, 가격은 거래량에 대한 관측치라는 것을 의미한다. 즉, 시간에 따른 가격의 관측은 그다지 좋은 샘플링 방법이 아니라는 것이다. 증권 가격과 거래량은 소형주이냐 대형주이냐, 우선주이냐 보통주이냐에 따라서 달라지기 때문에 당연히 증권의 종류마다 내포되어 있는 정보의 양과 흐름은 다르다. 여기서 증권가격의 상대적 시간 흐름을 유추해볼 수 있다.
