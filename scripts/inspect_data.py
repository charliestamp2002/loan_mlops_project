from loan_mlops.data.load_data import load_approval_data, load_default_data

def inspect_approval_basic_info(): 

    df = load_approval_data()
    print(" ---APPROVAL DATA---\n")
    print("First 5 rows:")
    print(df.head())

    print("\nFirst 10 rows:")
    print(df.head(10))

    print("\nLast 5 rows:")
    print(df.tail())

    print("\nDataset shape (rows, columns):")
    print(df.shape)

    print("\nDataset info (types, non-null counts, memory):")
    print(df.info())

    print("\nStatistical summary:")
    print(df.describe())

def inspect_default_basic_info():

    df = load_default_data()
    print(" ---DEFAULT DATA---\n")
    print("First 5 rows:")
    print(df.head())

    print("\nFirst 10 rows:")
    print(df.head(10))

    print("\nLast 5 rows:")
    print(df.tail())

    print("\nDataset shape (rows, columns):")
    print(df.shape)

    print("\nDataset info (types, non-null counts, memory):")
    print(df.info())

    print("\nStatistical summary:")
    print(df.describe())


def main(): 
    inspect_approval_basic_info()
    inspect_default_basic_info()

if __name__ == "__main__": 
    main()

