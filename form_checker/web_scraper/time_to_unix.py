import datetime

def convert_to_unix_timestamps(date_str):
    """Converts a date to a unix timestamp:
    
    Format: YYYY-MM-DD """
    # Convert the input date string to a datetime object
    date = datetime.datetime.strptime(date_str, '%Y-%m-%d')

    # Get the start and end of the day as datetime objects
    start_of_day = datetime.datetime(date.year, date.month, date.day, 0, 0, 0)
    # end_of_day = datetime.datetime(date.year, date.month, date.day, 23, 59, 59)

    # Convert the datetime objects to Unix timestamps (in seconds)
    start_timestamp = int(start_of_day.timestamp())
    # end_timestamp = int(end_of_day.timestamp())

    return start_timestamp

## example
# date_str = input("Enter a date (YYYY-MM-DD): ")
# start_timestamp, end_timestamp = convert_to_unix_timestamps(date_str)
# print(f"Start Timestamp: {start_timestamp}")
# print(f"End Timestamp: {end_timestamp}")