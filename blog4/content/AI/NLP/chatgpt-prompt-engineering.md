---
title: "ChatGPT 프롬프트 엔지니어링 가이드라인"
date: 2023-05-27 13:11:13
subtitle: "[DeepLearning.AI, OpenAI] ChatGPT Prompt Engineering for Developers"
category: ""
draft: false
---

> 이 글은 [ChatGPT Prompt Engineering for Developers]()강의를 정리한 글입니다.



**프롬프트 엔지니어링**이란 LLM(거대 언어 모델) 또는 생성 모델에 사용자가 의도하는 결과를 도출하도록 지시사항을 전달하는 작업이다. 프롬프트 엔지니어링은 생성 모델의 결과를 구체화하는 데에 초점이 맞추어져 있다. 따라서 모델에게 프롬프트를 제공하는 것은 아직 모델이 AGI(Artificial General Intelligence)와는 멀다는 것을 의미하며, 앞으로의 모델은 프롬프트를 예측하는 방향으로 나아갈 것이라고 예상된다. 그럼에도 불구하고, 프롬프트는 **자연어 지시사항**이고 모델은 이를 이해해서 많은 과제를 프롬프트에 따라 완수하기 때문에 모델이 "생각"하거나 "이해"한다고 지성을 가진 존재처럼 일컫게된다. 비록 Hallucination과 같은 LLM의 한계점이 존재하고 "생각"하기 보다는 대규모 데이터에 "적응"하는 것에 가깝지만, LLM이 이전에는 불가능했던 많은 과제를 수행할 수 있다는 점은 놀라우며 발전이 기대되는 분야라고 할 수 있다.

이 글에서는 프롬프트 엔지니어링에 적용할 수 있는 두가지 원칙과 원칙을 실행하는 전략을 예시와 함께 이해하고자 한다.

<br> 

### Setting
- `openai` 라이브러리의 `gpt-3.5-turbo model`을 활용한다.
- [Chat Completions endpoint](https://platform.openai.com/docs/guides/chat)에 따른다.

다음과 같이 설정할 수 있다.
```python
!pip install openai
```

```python
import openai
import os

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
# enviornment variable로 세팅한 경우
openai.api_key  = os.getenv('OPENAI_API_KEY') 
```

#### helper 함수를 세팅한다.
```python
def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]
```

<br>



## 원칙 1: 명확하고 구체적인 지시 사항을 전달한다.

### Tactic 1: 구획 문자(delimiters)를 사용하여 입력의 부분들을 분명하게 구분한다.

- 구획 문자는 다음과 같은 것을 활용할 수 있다: 
    - "```", 
    - """, 
    - ---, 
    - < >, 
    - XML Tags: <tag> </tag>, :

```python
text = f"""
You should express what you want a model to do by \ 
providing instructions that are as clear and \ 
specific as you can possibly make them. \ 
This will guide the model towards the desired output, \ 
and reduce the chances of receiving irrelevant \ 
or incorrect responses. Don't confuse writing a \ 
clear prompt with writing a short prompt. \ 
In many cases, longer prompts provide more clarity \ 
and context for the model, which can lead to \ 
more detailed and relevant outputs.
"""
prompt = f"""
Summarize the text delimited by triple backticks \ 
into a single sentence.
```{text}```
"""
response = get_completion(prompt)
print(response)
```

### Tactic 2: 구조화된 아웃풋을 요구한다.
- JSON, HTML

```python
prompt = f"""
Generate a list of three made-up book titles along \ 
with their authors and genres. 
Provide them in JSON format with the following keys: 
book_id, title, author, genre.
"""
response = get_completion(prompt)
print(response)
```

### Tactic 3: 모델에게 결과 도출 조건들이 충족되었는지 확인하도록 지시한다. 

```python
text_1 = f"""
Making a cup of tea is easy! First, you need to get some \ 
water boiling. While that's happening, \ 
grab a cup and put a tea bag in it. Once the water is \ 
hot enough, just pour it over the tea bag. \ 
Let it sit for a bit so the tea can steep. After a \ 
few minutes, take out the tea bag. If you \ 
like, you can add some sugar or milk to taste. \ 
And that's it! You've got yourself a delicious \ 
cup of tea to enjoy.
"""
prompt = f"""
You will be provided with text delimited by triple quotes. 
If it contains a sequence of instructions, \ 
re-write those instructions in the following format:

Step 1 - ...
Step 2 - …
…
Step N - …

If the text does not contain a sequence of instructions, \ 
then simply write \"No steps provided.\"

\"\"\"{text_1}\"\"\"
"""
response = get_completion(prompt)
print("Completion for Text 1:")
print(response)
```

```
Clear and specific instructions should be provided to guide a model towards the desired output, and longer prompts can provide more clarity and context for the model, leading to more detailed and relevant outputs.
```


### Tactic 4: "Few-shot" 프롬프팅
- 원하는 과제를 성공적으로 수행한 예시를 제공해서 모델이 사용자 의도에 부합하는 결과를 제출하는 것을 유도할 수 있다. 

```python
prompt = f"""
Your task is to answer in a consistent style.

<child>: Teach me about patience.

<grandparent>: The river that carves the deepest \ 
valley flows from a modest spring; the \ 
grandest symphony originates from a single note; \ 
the most intricate tapestry begins with a solitary thread.

<child>: Teach me about resilience.
"""
response = get_completion(prompt)
print(response)
```

```
<grandparent>: Resilience is like a tree that bends with the wind but never breaks. It is the ability to bounce back from adversity and keep moving forward, even when things get tough. Just like a tree that grows stronger with each storm it weathers, resilience is a quality that can be developed and strengthened over time.
```

<br>

## 원칙 2: 모델이 "생각"할 시간을 준다.

### Tactic 1: 과제를 완료하기 위한 단계를 구체화한다.

```python
text = f"""
In a charming village, siblings Jack and Jill set out on \ 
a quest to fetch water from a hilltop \ 
well. As they climbed, singing joyfully, misfortune \ 
struck—Jack tripped on a stone and tumbled \ 
down the hill, with Jill following suit. \ 
Though slightly battered, the pair returned home to \ 
comforting embraces. Despite the mishap, \ 
their adventurous spirits remained undimmed, and they \ 
continued exploring with delight.
"""
# example 1
prompt_1 = f"""
Perform the following actions: 
1 - Summarize the following text delimited by triple \
backticks with 1 sentence.
2 - Translate the summary into French.
3 - List each name in the French summary.
4 - Output a json object that contains the following \
keys: french_summary, num_names.

Separate your answers with line breaks.

Text:
```{text}```
"""
response = get_completion(prompt_1)
print("Completion for prompt 1:")
print(response)
```

```
Completion for prompt 1:
Two siblings, Jack and Jill, go on a quest to fetch water from a well on a hilltop, but misfortune strikes and they both tumble down the hill, returning home slightly battered but with their adventurous spirits undimmed.

Deux frères et sœurs, Jack et Jill, partent en quête d'eau d'un puits sur une colline, mais un malheur frappe et ils tombent tous les deux de la colline, rentrant chez eux légèrement meurtris mais avec leurs esprits aventureux intacts. 
Noms: Jack, Jill. 

{
  "french_summary": "Deux frères et sœurs, Jack et Jill, partent en quête d'eau d'un puits sur une colline, mais un malheur frappe et ils tombent tous les deux de la colline, rentrant chez eux légèrement meurtris mais avec leurs esprits aventureux intacts.",
  "num_names": 2
}
```

#### 구체적인 포맷으로 결과를 도출하도록 요구하자.

```python
prompt_2 = f"""
Your task is to perform the following actions: 
1 - Summarize the following text delimited by 
  <> with 1 sentence.
2 - Translate the summary into French.
3 - List each name in the French summary.
4 - Output a json object that contains the 
  following keys: french_summary, num_names.

Use the following format:
Text: <text to summarize>
Summary: <summary>
Translation: <summary translation>
Names: <list of names in Italian summary>
Output JSON: <json with summary and num_names>

Text: <{text}>
"""
response = get_completion(prompt_2)
print("\nCompletion for prompt 2:")
print(response)
```

```
Completion for prompt 2:
Summary: Jack and Jill go on a quest to fetch water, but misfortune strikes and they tumble down the hill, returning home slightly battered but with their adventurous spirits undimmed. 
Translation: Jack et Jill partent en quête d'eau, mais un malheur frappe et ils tombent de la colline, rentrant chez eux légèrement meurtris mais avec leurs esprits aventureux intacts.
Names: Jack, Jill
Output JSON: {"french_summary": "Jack et Jill partent en quête d'eau, mais un malheur frappe et ils tombent de la colline, rentrant chez eux légèrement meurtris mais avec leurs esprits aventureux intacts.", "num_names": 2}
```

### Tactic 2: 모델이 결과를 도출하기 전에, 먼저 스스로 답을 생성하고 이를 결과와 비교해 답을 내리도록 요구한다.


예를 들어 다음과 같이 학생의 답이 맞는지 검증하도록 해보자.

```python
prompt = f"""
Determine if the student's solution is correct or not.

Question:
I'm building a solar power installation and I need \
 help working out the financials. 
- Land costs $100 / square foot
- I can buy solar panels for $250 / square foot
- I negotiated a contract for maintenance that will cost \ 
me a flat $100k per year, and an additional $10 / square \
foot
What is the total cost for the first year of operations 
as a function of the number of square feet.

Student's Solution:
Let x be the size of the installation in square feet.
Costs:
1. Land cost: 100x
2. Solar panel cost: 250x
3. Maintenance cost: 100,000 + 100x
Total cost: 100x + 250x + 100,000 + 100x = 450x + 100,000
"""
response = get_completion(prompt)
print(response)
```

```python
The student's solution is correct.
```

위에서 학생의 답에 `3.`을 보면 `10x` 대신 `100x`를 써 답이 틀렸지만, 모델은 이 답을 맞다고 결론지었다. 

#### 모델이 먼저 자신의 답을 생성해서 이와 학생의 답안을 비교하여 결론내리도록 하면 문제를 해결할 수 있다.

```python
prompt = f"""
Your task is to determine if the student's solution \
is correct or not.
To solve the problem do the following:
- First, work out your own solution to the problem. 
- Then compare your solution to the student's solution \ 
and evaluate if the student's solution is correct or not. 
Don't decide if the student's solution is correct until 
you have done the problem yourself.

Use the following format:
Question:
"""
question here
"""
Student's solution:
"""
student's solution here
"""
Actual solution:
"""
steps to work out the solution and your solution here
"""
Is the student's solution the same as actual solution \
just calculated:
"""
yes or no
"""
Student grade:
"""
correct or incorrect
"""

Question:
"""
I'm building a solar power installation and I need help \
working out the financials. 
- Land costs $100 / square foot
- I can buy solar panels for $250 / square foot
- I negotiated a contract for maintenance that will cost \
me a flat $100k per year, and an additional $10 / square \
foot
What is the total cost for the first year of operations \
as a function of the number of square feet.
"""
Student's solution:
"""
Let x be the size of the installation in square feet.
Costs:
1. Land cost: 100x
2. Solar panel cost: 250x
3. Maintenance cost: 100,000 + 100x
Total cost: 100x + 250x + 100,000 + 100x = 450x + 100,000
"""
Actual solution:
"""
response = get_completion(prompt)
print(response)
```

```python
Let x be the size of the installation in square feet.

Costs:
1. Land cost: 100x
2. Solar panel cost: 250x
3. Maintenance cost: 100,000 + 10x

Total cost: 100x + 250x + 100,000 + 10x = 360x + 100,000

Is the student's solution the same as actual solution just calculated:
No

Student grade:
Incorrect
```

`3.`에서 맞은 답을 도출하고 학생의 답을 틀렸다고 결론지은 것을 확인할 수 있다.

<br>

## 모델의 한계점: Hallucinations
- 모델은 학습과정에서 많은 데이터에 노출되지만 학습 과정 중에 겪은 정보를 모두 "저장"하고 있지는 않기 때문에, 모델 자신이 가지고 있는 지식의 경계를 잘 알지 못한다. 따라서 모델은 **경계가 모호한 질문**에 대해서도 답을 생성하고, 이 대답이 그럴듯하게 보이지만 사실이 아닐 수 있다는 점에 유의해야 한다.
- 아래 예시에서 `Boie`는 실제 존재하는 칫솔 회사인데, 질문에서 있다고 가정하고 설명을 요구한 `AeroGlide UltraSlim Smart Toothbrush`는 존재하지 않는 상품이다. 그럼에도 불구하고 모델이 그럴듯한 답안을 생성해내는 것을 확인할 수 있다.

```python
prompt = f"""
Tell me about AeroGlide UltraSlim Smart Toothbrush by Boie
"""
response = get_completion(prompt)
print(response)
```

```
The AeroGlide UltraSlim Smart Toothbrush by Boie is a high-tech toothbrush that uses advanced sonic technology to provide a deep and thorough clean. It features a slim and sleek design that makes it easy to hold and maneuver, and it comes with a range of smart features that help you optimize your brushing routine.

One of the key features of the AeroGlide UltraSlim Smart Toothbrush is its advanced sonic technology, which uses high-frequency vibrations to break up plaque and bacteria on your teeth and gums. This technology is highly effective at removing even the toughest stains and buildup, leaving your teeth feeling clean and refreshed.

In addition to its sonic technology, the AeroGlide UltraSlim Smart Toothbrush also comes with a range of smart features that help you optimize your brushing routine. These include a built-in timer that ensures you brush for the recommended two minutes, as well as a pressure sensor that alerts you if you're brushing too hard.

Overall, the AeroGlide UltraSlim Smart Toothbrush by Boie is a highly advanced and effective toothbrush that is perfect for anyone looking to take their oral hygiene to the next level. With its advanced sonic technology and smart features, it provides a deep and thorough clean that leaves your teeth feeling fresh and healthy.
```

#### Hallucination을 줄이는 방법은: 1. 먼저 관련된 정보를 찾고 2. 이 정보에 기반해서 답을 생성하는 것이다.

