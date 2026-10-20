import json
import random
from datetime import datetime

from colorama import Fore, Style, init

init(autoreset=True)

RULES = {
    "hello": ["Hi there!", "Hello! How can I help?"],
    "how are you": ["I'm just code, but I'm running fine!"],
    "inference": ["Inference means using a trained model to make predictions."],
    "training": ["Training is when a model learns its weights from data."],
}


class ChatBot:
    def __init__(self, name):
        self.name = name
        self.history = []

    def reply(self, message):
        text = message.lower()
        if "time" in text:
            return "It's " + datetime.now().strftime("%H:%M")
        for keyword, answers in RULES.items():
            if keyword in text:
                return random.choice(answers)
        return "Sorry, I don't know that yet. Type /help."

    def save_history(self, filename="history.json"):
        with open(filename, "w") as f:
            json.dump(self.history, f, indent=2)


def main():
    bot = ChatBot("InferBot")
    print(Fore.CYAN + f"Hi, I'm {bot.name}! Type /help for commands, /quit to exit.")
    while True:
        user = input(Fore.GREEN + "You: " + Style.RESET_ALL).strip()
        if not user:
            continue
        if user == "/quit":
            bot.save_history()
            print(Fore.CYAN + "Bye! Chat saved to history.json")
            break
        if user == "/help":
            print("Try: hello, how are you, inference, training, time")
            continue
        answer = bot.reply(user)
        bot.history.append({"you": user, "bot": answer})
        print(Fore.YELLOW + f"{bot.name}: " + Style.RESET_ALL + answer)


if __name__ == "__main__":
    main()