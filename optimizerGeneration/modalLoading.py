import modal


app = modal.App("optimizerBuild")

image = modal.Image.debian_slim().pip_install("groq", "dotenv", "datetime")

sandBoxImage = modal.Image.debian_slim().pip_install(
    "groq", "dotenv", "datetime", "torch", "numpy"
)

volume = modal.Volume.from_name("benchmark-responses")

optimizersVolume = modal.Volume.from_name("optimizers")
optimizerSysPromptVolume = modal.Volume.from_name("optimizerSysPrompt")


sandboxApp = modal.App.lookup("sandboxedExecution", create_if_missing=True)
sb = modal.Sandbox.create(app=sandboxApp, image=sandBoxImage)


@app.function(
    gpu="A100",
    timeout=3600,
    image=image,
    volumes={"/optimizers": optimizersVolume, "/sysPrompt": optimizerSysPromptVolume},
    secrets=[modal.Secret.from_name("optimizerSecret")],
)
def sendCode():
    import datetime
    import os
    from groq import Groq
    from dotenv import load_dotenv
    import time

    def generate_optimizer():
        with open("/sysPrompt/OPTIMIZER_SYSTEM_PROMPT.txt", "r") as file:
            optimizer_system_prompt = file.read()

        load_dotenv()

        client = Groq(
            api_key=os.environ["GROQ_KEY"],
        )

        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": optimizer_system_prompt,
                }
            ],
            model="llama-3.3-70b-versatile",
        )

        return chat_completion.choices[0].message.content

    optimizerCode = generate_optimizer()
    optimizerCode = optimizerCode.replace("```python", "").replace("```", "")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    optimizer_file_name = f"optimizer_{timestamp}.py"
    optimizer_file_path = os.path.join("/optimizers", optimizer_file_name)
    with open(optimizer_file_path, "w") as f:
        f.write(optimizerCode)
    optimizersVolume.commit()
    print("Committed to volume")
    print(optimizer_file_name)
    with sb.open(optimizer_file_name, "w") as g:
        g.write(optimizerCode)
    s = sb.open(optimizer_file_name, "rb")
    print(s.read())

    time.sleep(1)
    p = sb.exec("python", optimizer_file_name)
    print(p.stdout.read())
    print("--------------------------------")
    print(p.stderr.read())
    sb.terminate()


@app.local_entrypoint()
def main():
    sendCode.remote()


