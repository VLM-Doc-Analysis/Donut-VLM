# Donut CORD-v2 영수증 파싱 — 토큰 종류 가이드

> Donut(CORD-v2) 영수증 파싱이 출력하는 **토큰의 종류와 의미**를 정리한 문서.
> 실습 노트북: [`donut_CORD_v2_fine_tunned_test_kardi.ipynb`](donut_CORD_v2_fine_tunned_test_kardi.ipynb)
> 예시 이미지: [`test_data/CORD_Test_Data.png`](test_data/CORD_Test_Data.png)

Donut은 영수증을 **JSON이 아니라 XML 스타일 토큰 시퀀스**로 생성하고, 후처리(`token2json`)로 JSON으로 되돌린다.
토큰은 **두 층위**로 나뉜다 — **① 구조(틀)** + **② 필드(내용)**.

```
<s_cord-v2>  <s_total>12500</s_total> …  </s>
└ task_prompt ┘ └─────── ② 필드 토큰 ───────┘ └ eos ┘
   (① 시작)                                     (① 끝)
```

---

## 1. ① 구조 / 특수 토큰 — "틀"

영수증 내용과 무관하게 **시퀀스의 뼈대**를 만든다.

| 토큰 | 역할 |
|---|---|
| `<s_cord-v2>` | **task prompt = 디코더 시작 토큰.** "지금부터 영수증 JSON을 토큰으로 뱉어라" 신호. CORD-v2 전용으로 `add_special_tokens` 등록됨 |
| `</s>` | **eos** — 생성 종료 |
| `<s>` | bos (문장 시작) |
| `<pad>` | 패딩 (빈자리 채움, 학습 시 loss 무시) |
| `<unk>` | 사전에 없는 글자 |
| `<mask>` | 마스킹용 (사전학습 잔재, 추론엔 거의 안 씀) |
| `<sep/>` | **리스트 구분자** — 메뉴가 여러 개일 때 항목 사이를 나눔 |

---

## 2. ② 필드(스키마) 토큰 — "내용"

의미 항목을 `<s_키>값</s_키>` 쌍으로 표현한다. CORD-v2는 영수증을 **4개 그룹**으로 본다.

| 그룹(상위) | 의미 | 대표 하위 필드 |
|---|---|---|
| **`menu`** | 주문 항목(리스트) | `nm`(메뉴명) · `cnt`(수량) · `unitprice`(단가) · `price`(금액) · `discountprice` · `num` · `sub_nm`/`sub_price` |
| **`sub_total`** | 소계 영역 | `subtotal_price`(소계) · `discount_price`(할인) · `service_price`(봉사료) · `tax_price`(세금) · `etc` |
| **`total`** | 합계 영역 | `total_price`(총액) · `cashprice`(현금) · `changeprice`(거스름돈) · `creditcardprice`(카드) · `menuqty_cnt`(총수량) |
| **`void_menu`** | 취소된 항목 | `nm` · `price` |

> 이 필드 토큰들은 학습 때 라벨에 등장한 키를 **모두 `add_special_tokens`로 등록**해 디코더가 만들어낼 수 있게 한 것이다.

---

## 3. `menu` 그룹 — 하위 필드 상세

`menu`는 **주문 항목 목록**. 항목이 여러 개면 `<sep/>`로 구분된 **리스트**가 되고, 각 항목은 아래 필드의 조합이다.

### (a) 항목 본체 필드 — 메뉴 한 줄의 정보

| 필드 | 의미 | 예시 | 비고 |
|---|---|---|---|
| `nm` | **메뉴 이름** | `아메리카노` | 거의 항상 등장(핵심) |
| `cnt` | **수량** | `2` | 핵심 |
| `unitprice` | **단가**(개당 가격) | `4,500` | 단가가 찍힐 때 |
| `price` | **금액**(항목 합계) | `9,000` | 핵심 (보통 단가×수량) |
| `num` | 메뉴 **번호/코드** | `101` | POS 항목번호 찍히는 영수증 |
| `discountprice` | 항목 **할인 금액** | `-1,000` | 항목별 할인이 있을 때 |
| `itemsubtotal` | 항목 **소계** | `8,000` | 할인 반영 소계가 별도로 찍힐 때 |
| `vatyn` | **부가세 대상 여부**(Y/N) | `Y` | 과세/면세 표시 영수증 |
| `etc` | **기타** 부가 정보 | `포장`, `ICE` | 잡다한 표기 |

### (b) 세부(`sub_`) 필드 — 옵션 / 세트 구성

한 메뉴 **아래 딸린 하위 항목**(세트 구성품·토핑·옵션). 본체 필드와 이름만 `sub_`가 붙은 짝이며, 본체 안에 **중첩**된다.

| 필드 | 의미 | 예시 |
|---|---|---|
| `sub_nm` | 옵션/구성품 **이름** | `샷 추가` |
| `sub_cnt` | 옵션 **수량** | `1` |
| `sub_unitprice` | 옵션 **단가** | `500` |
| `sub_price` | 옵션 **금액** | `500` |
| `sub_etc` | 옵션 **기타** | `라지` |

> **왜 `sub_`가 필요한가** — "세트 메뉴 + 구성품", "음료 + 샷 추가"처럼 한 항목에 하위 항목이 들여쓰기로 붙는 구조를 표현하기 위해.

---

## 4. 실제 예시 — `test_data/CORD_Test_Data.png`

![CORD 테스트 영수증](test_data/CORD_Test_Data.png)

**Auntie Anne's (인도네시아, 단위 IDR)** 영수증. 읽히는 내용:

```
Auntie Anne's
CINNAMON SUGAR    1 x 17.000    17.000
SUB TOTAL                        17.000
GRAND TOTAL                      17.000
CASH IDR                         20.000
CHANGE DUE                        3.000
```
> `17.000` = 인도네시아 표기로 17,000 IDR (점이 천 단위 구분).

### menu 매핑 — 이 영수증은 본체 필드 4개만 사용

`CINNAMON SUGAR  1 x 17.000  17.000` 한 줄 = **menu 항목 1개**.

| menu 필드 | 값 | 근거 |
|---|---|---|
| `nm` | `CINNAMON SUGAR` | 항목 이름 |
| `cnt` | `1` | `1 x` 의 `1` |
| `unitprice` | `17.000` | `1 x 17.000` 의 단가 |
| `price` | `17.000` | 줄 끝 합계 |

> 📌 이 영수증엔 **`sub_*` 가 없다**(옵션·세트 없음). `discountprice`·`num`·`vatyn`도 안 찍혀서 안 나온다.
> 만약 아래에 `+ Extra Sugar  500` 줄이 있었다면 → `sub_nm:"Extra Sugar"`, `sub_price:"500"`로 중첩됐을 것.

### 전체 토큰 시퀀스

```
<s_cord-v2>
  <s_menu>
    <s_nm>CINNAMON SUGAR</s_nm>
    <s_cnt>1</s_cnt>
    <s_unitprice>17.000</s_unitprice>
    <s_price>17.000</s_price>
  </s_menu>
  <s_sub_total>
    <s_subtotal_price>17.000</s_subtotal_price>
  </s_sub_total>
  <s_total>
    <s_total_price>17.000</s_total_price>
    <s_cashprice>20.000</s_cashprice>
    <s_changeprice>3.000</s_changeprice>
  </s_total>
</s>
```

### `token2json` 복원 결과

```json
{
  "menu": {
    "nm": "CINNAMON SUGAR",
    "cnt": "1",
    "unitprice": "17.000",
    "price": "17.000"
  },
  "sub_total": { "subtotal_price": "17.000" },
  "total": {
    "total_price": "17.000",
    "cashprice": "20.000",
    "changeprice": "3.000"
  }
}
```

---

## 5. 메뉴가 여러 개일 때 — `<sep/>`

```
<s_menu>
  <s_nm>아메리카노</s_nm><s_cnt>2</s_cnt><s_price>9,000</s_price>
<sep/>
  <s_nm>카페라떼</s_nm><s_cnt>1</s_cnt><s_price>5,000</s_price>
</s_menu>
```
→ `"menu"`가 **리스트**가 된다:
```json
{ "menu": [
    {"nm":"아메리카노","cnt":"2","price":"9,000"},
    {"nm":"카페라떼","cnt":"1","price":"5,000"}
] }
```

---

## 6. 정리 & 주의

- **틀(①)**: `<s_cord-v2>`(시작) … `</s>`(끝), 리스트는 `<sep/>`.
- **내용(②)**: `menu / sub_total / total / void_menu` 4그룹 + 하위 필드.
- **menu 핵심 필드**: `nm`, `cnt`, `price` (+단가 있으면 `unitprice`). 나머지는 영수증 양식에 따라 선택.
- **`sub_*`** = 옵션/세트 구성품(중첩), 본체 필드와 1:1 대응.
- 규칙: **`task_prompt + <s_키>값</s_키>… + eos`** → `token2json`으로 JSON 복원.
- ⚠️ 어떤 필드가 실제로 나올지는 **학습 라벨에 등장한 키 집합**으로 결정된다 — 표준 스키마라도 모델이 라벨로 배우지 않은 키는 못 뱉는다.

> 대표 하위 필드만 실었다. CORD-v2 전체 스키마엔 `service_price/othersvc_price/emoneyprice/menutype_cnt` 등 더 많은 키가 있다.
