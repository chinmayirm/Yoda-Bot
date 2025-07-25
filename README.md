A Jedi-style introspective AI guide powered by Master Yoda's authentic dialogue from the Star Wars franchise as well as rule based grammar. This chatbot helps users reflect on their personality, values, habits, and dreams while responding in Yoda's iconic speech.
# Yoda Bot Chat UI

A minimal conversational AI that responds like Master Yoda. Built with Streamlit and OpenAI's GPT API, this chatbot transforms your everyday queries into Yoda-style responses. Built to encourage self-reflection, humor, and introspection through a fun interface.

## Example Screenshots


![WhatsApp Image 2025-07-23 at 13 38 15_3bda0d4d](https://github.com/user-attachments/assets/397348e9-100f-4a4e-9533-f90d140808d1)



![WhatsApp Image 2025-07-23 at 13 40 06_e04d8958](https://github.com/user-attachments/assets/fdbe3780-1d8f-4f24-8cd2-03eba68cab9a)




## Features

- Yoda-style grammar conversion using prompt engineering
- Interactive chat UI using Streamlit
- Real-time responses powered by OpenAI GPT
- Local image avatar support (Yoda's face)
- Easy-to-deploy on any system with internet access

## Why Yoda?

Master Yoda often places the verb and subject at the end of the sentence, offering wisdom in a cryptic but thought-provoking form. This design promotes a unique way of reflecting on user inputs. Think of it as introspection, guided by a Jedi.

Also just because I am a massive Star Wars fan.
## Tech Stack

| Component     | Technology           |
|---------------|----------------------|
| UI Framework  | Streamlit            |
| Backend Logic | Python 3.11          |
| LLM           | DialoGPT-small       |
| Avatar Image  | Custom-uploaded PNG  |
| Hosting       | Google Colab / Local |

Datasets used: Cornell Movie Dialogues Corpus
(https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html)

## Setup Instructions

1. Clone the repository or download the script.
2. Open the Python script `yoda_chatbot_app.py`.
3. Install required dependencies:
   ```bash
   pip install streamlit openai
   ```
4. Launch the app:
   ```bash
   streamlit run yoda_chatbot_app.py
   ```

5. Upload your own `yoda.png` image in the root directory if needed.




## Sample Interaction

> **You:** I'm excited about the upcoming Superman movie but I'm worried I might not be able to watch it this week.  
> **Master Yoda:** Think you're right, yeah I, hmmm.  

> **You:** What advice do you have for me today?  
> **Master Yoda:** Jedi here, I can't have any of my own advice, I'm a, hmmm.

> **You:** Who is your favourite Naruto character?  
> **Master Yoda:** Your favourite Naruto character, who is, hmmm.

## Benefits of Using This Bot

- Encourages **self-introspection** via altered sentence structure  
- Adds **fun** to daily interactions  
- Useful for **demoing prompt engineering** techniques  
- Lightweight and customizable chatbot shell

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Author

Built by Chinmayi R M just for fun ;)
P.S. Still working on some tweaks
