from src.chatbot import get_response

def main():
    print("Welcome to Eva Personal Assistant!")
    while True:
        message = input("You: ")
        response = get_response(message)
        print(f"Bot: {response}")

if __name__ == "__main__":
    main()
