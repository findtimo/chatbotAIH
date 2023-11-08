
# HealthServe Telegram Chatbot AIH

Healthserve chatbot made for SMU AIH COR3301.






## Deployment

To deploy this project. The project is split into 4 files. Remember to include environment variables.

**generateVectorStore.py**

Precreate the vector stores to provide contextual insights for the LLM when it runs. To start, put relevant documents into ./docs/{files.pdf} here

**main.py:**

Provides the telegram chatbot commands.
Inside are features such as google translation, IBM speech to text, and sentiment analysis. It uses OpenAI's gpt3.5 turbo to evaluate and provide responses. There is a custom prompt provided which can be altered to narrow results.

## Evaluation & Testing
To evaluate the test the chatbot's accuracy, helpfulness, etc.

**createDataset.py**

We use LangSmith APIs to evaluate the chatbot. This python file is to first create dummy data for LangSmith to work with. Provides a "question" and "expected_ans" section.

**read.py**

For LangSmith to evaluate the test data created before, and provide the mean, median, mode for helpfulness, correctness, context accuracy, and harmfulness.

## Final Thoughts
I ran test cases and put it in an excel sheet for anyone interested to see. I think its a good fun mini project.
## Tech Stack

**Client:** Python

**LLM:** OpenAI, LangSmith, LangChain, GoogleTranslate, IBM


## Authors

- [@findtimo](https://www.github.com/findtimo)

