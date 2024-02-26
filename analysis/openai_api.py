import openai

openai.api_key = ''


def get_email():
    messages = [
        {"role": "system", "content": "Lorsque je dis: 0 donne un exemple email complet considéré comme spam, 1 si normal. "
                                      "Répondre en français, remplacer tous les champs par des données qui "
                                      "existent, objet et contenu uniquement. "
                                      "Ne pas donner d'information supplémentaires au début ni à la fin."},
        {"role": "user", "content": "0"}
    ]

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        stream=False,
    )

    return completion.choices[0].message.content


def append_on_csv(csv_path, content):
    with open(csv_path, 'a') as f:
        f.write(format_into_one_line(content) + '\n')


def format_into_one_line(content):
    return content.replace('\n', ' ').replace('\r', ' ')


if __name__ == '__main__':
    for i in range(50):
        print(i)
        email = get_email()
        append_on_csv('data/spam.csv', email)
