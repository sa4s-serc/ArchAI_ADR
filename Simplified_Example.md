## Simplified way to generate a single Design Decision

To generate Design Decisions using GPT3 and above models, one needs to call the OpenAI api.
Calling the OpenAI api will cost time as well as money (api credits).
Hence to verify the functionality of the work in a very short time and cost effectively, one can visit OpenAI playground and generate design decision themselves. The steps to do that are as follows:

 - Visit https://platform.openai.com/playground
 - Select Complete to use text completion models (GPT-3 ada, davinci, and GPT-3.5 text-davinci-003). Select Chat for using chat models (GPT-3.5 turbo and GPT-4). Then select the model.
 - Use the required prompt template as given in the paper. For 0-shot approach use the prompt given in Figure 3 and 4 for completion and chat models respectively. For few-shot approach use the prompt given in Figure 5 and 5 for completion and chat models respectively.
 - The models response would be the expected Design Decision.

One may give a simple context like : "We need to decide on whether to use Python as a programming language for our project. Our project involves data analysis, machine learning, and web development." to generate the required Design Decision.
