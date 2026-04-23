from process_resume import ResumeProcessor

async def process_resume():
    resume_processor = ResumeProcessor()
    await resume_processor.process_resume()


def main():
    print("Hello from interview-prep!")


if __name__ == "__main__":
    main()
