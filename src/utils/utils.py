import io
import json
import pandas as pd


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def add_eos_text(example, tokenizer):
    eos_token = tokenizer.eos_token
    example["text"] = [ex + " " + eos_token for ex in example["text"]]
    return example


def group_texts(examples, block_size=2048):
    # Concatenate all texts.
    # print(examples)
    # return
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


def parse_chat(fname, p1="Deepak", p2="Priya"):
    """Parse whatsapp chat between p1 and p2 and output a dataframe with two columns, user and message"""
    chats = {"User": [], "message": []}

    with open(fname, "r") as f:
        msg_d = ""
        msg_a = ""
        for line in f:
            if p1 + ":" in line:
                if msg_a != "":
                    chats["User"].append(p2)
                    chats["message"].append(msg_a)
                msg_d = msg_d + line.split(p1 + ":")[1]
                msg_a = ""
            if p2 + ":" in line:
                if msg_d != "":
                    chats["User"].append(p1)
                    chats["message"].append(msg_d)
                msg_d = ""
                msg_a = msg_a + line.split(p2 + ":")[1]

    # print(chats)
    df = pd.DataFrame.from_dict(chats)
    df["message"] = df["message"].str.replace("\n", ",")
    df["message"] = df["message"].str.replace(",\s*$", "", regex=True)
    df["message"] = df["message"].apply(
        lambda x: x.encode("ascii", "ignore").decode("ascii")
    )

    return df


def parse_clean_chat(fname, fname_out="chat_history.txt", p1="Deepak", p2="Priya"):
    """Parse WhatsApp chat between p1 and p2 and write a text file"""
    df = parse_chat(fname, p1, p2)
    df["message"] = df["User"] + ": " + df["message"]
    with open(fname_out, "w") as f:
        for msg in df["message"].to_list():
            f.write(msg)
            f.write("\n")

    f.close()


def clean_chat(fname, fname_out):
    line_count = 0
    f_out = open(fname_out, "w")
    with open(fname, "r") as f:
        for line in f:
            line = line.encode("ascii", "ignore").decode("ascii")
            split_line = line.split("]")
            try:
                print(split_line[1])
                f_out.write(split_line[1])
                line_count = line_count + 1
            except:
                continue
    f_out.close()


def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f


def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict
