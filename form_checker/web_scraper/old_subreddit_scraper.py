import requests
import os
from time_to_unix import convert_to_unix_timestamps

class subreddit_scraper():

    def __init__(self,
                 subreddit='formcheck',
                 tag='Deadlift',
                 ):
        """ Initialize our deadlift scraper tool.
        Pulls from Reddit's r/formcheck subreddit under the 
        'Deadlift' tag."""
        self.subreddit = subreddit
        self.tag = tag

        # Create a directory to save the downloaded GIFs
        self.save_directory = f'web_scraper/reddit/{self.subreddit}/{self.tag}/imgur_gifs'


    def _scrape(self,
                start_time="2022-01-01",
                end_time="2022-12-30",):

        # Convert times given to Unix timestamps (in seconds)
        self.start_time = convert_to_unix_timestamps(start_time)
        self.end_time = convert_to_unix_timestamps(end_time)


        if not os.path.exists(self.save_directory):
            os.makedirs(self.save_directory)

        # Define the Pushshift API URL
        url = f'https://api.pushshift.io/reddit/search/submission/?subreddit={self.subreddit}&q={self.tag}&after={self.start_time}&before={self.end_time}&limit=100'

        # Send a GET request to the API
        response = requests.get(url)

        # Extract the JSON data from the response
        data = response.json()

        # Iterate over the posts and download the GIFs
        for post in data['data']:
            url = post['url']
            if 'imgur' in url and 'gifv' in url:
                gif_url = url.replace('gifv', 'gif')
                gif_filename = gif_url.split('/')[-1]
                gif_path = os.path.join(self.save_directory, gif_filename)

                # Download the GIF
                gif_response = requests.get(gif_url)
                with open(gif_path, 'wb') as f:
                    f.write(gif_response.content)
                print(f'Downloaded: {gif_filename}')

        print('Scraping completed!')

if __name__ == '__main__':
    scraper = subreddit_scraper()

    scraper._scrape()