import pypdf
import os


def main(directory):

    block_pages = 0
    for manual in os.listdir(directory):
        pdf_path = os.path.join(directory, manual)
        print(pdf_path)
        with open(pdf_path, 'rb') as file:
            reader = pypdf.PdfReader(file)
            text = ''
            block_pages = 0
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text += page.extract_text()

                if page_num % 85 == 0:
                    filename = "./data/manuals/extracted/"+manual+"_block"+str(block_pages)+".txt"
                    with open(filename, "w") as text_manual:
                        text_manual.write(text)
                    block_pages += 1
                    text = ''

if __name__== "__main__":
    directory = 'data/manuals/raw'
    main(directory)
