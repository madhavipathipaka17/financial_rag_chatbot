from src.chains import ask_question


test_cases = [
    {
        "question": "What was NVIDIA's revenue in fiscal year 2024?",
        "expected": "$60,922 million"
    },
    {
        "question": "What was NVIDIA's net income in fiscal year 2024?",
        "expected": "$29,760 million"
    },
    {
        "question": "What was NVIDIA's Data Center revenue in fiscal year 2024?",
        "expected": "$47,525 million"
    },
    {
        "question": "What was NVIDIA's Data Center revenue in fiscal year 2020?",
        "expected": "$2,983 million"
    },
    {
        "question": "What are some major risk factors mentioned by NVIDIA?",
        "expected": "Relevant NVIDIA risk factors"
    }
]


passed = 0
failed = 0


for i, test in enumerate(test_cases, start=1):

    print("\n" + "=" * 60)
    print(f"TEST {i}")
    print("=" * 60)

    print("Question:")
    print(test["question"])

    answer, documents = ask_question(test["question"])

    print("\nExpected:")
    print(test["expected"])

    print("\nGenerated Answer:")
    print(answer)

    # Check result
    if test["question"].startswith("What are some major risk factors"):
        # For this question, check whether the answer contains
        # some important risk-factor keywords.
        keywords = ["competition", "market"]
        result = all(
            keyword.lower() in answer.lower()
            for keyword in keywords
        )
    else:
        # For numerical questions, check whether
        # the expected value appears in the generated answer.
        result = test["expected"].lower() in answer.lower()

    if result:
        print("\nResult: PASS")
        passed += 1
    else:
        print("\nResult: FAIL")
        failed += 1

    print("\nRetrieved Sources:")

    for document in documents:
        print(
            f"- {document.metadata.get('source')} | "
            f"Page: {document.metadata.get('page')}"
        )


# Final evaluation summary
total = len(test_cases)
accuracy = (passed / total) * 100

print("\n" + "=" * 60)
print("EVALUATION SUMMARY")
print("=" * 60)

print(f"Total Tests: {total}")
print(f"Passed: {passed}")
print(f"Failed: {failed}")
print(f"Accuracy: {accuracy:.1f}%")