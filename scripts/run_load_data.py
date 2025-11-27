from loan_mlops.data.load_data import load_approval_data, load_default_data

def main() -> None:
    approval_df = load_approval_data()
    default_df = load_default_data()

    print("Approval data head:")
    print(approval_df.head())
    print("\nDefault data head:")
    print(default_df.head())


if __name__ == "__main__":
    main()