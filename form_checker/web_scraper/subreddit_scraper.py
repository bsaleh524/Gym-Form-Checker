import requests
import praw
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
        # Define the subreddit and tag to scrape
        self.subreddit = subreddit
        self.tag = tag

        # Reddit API credentials
        self.client_id = 'ccv85vrwQOcalpGKi8rAEg'
        self.client_secret = 'WR22NH801cC8uEwMPck5lDprOqF7-A'
        self.user_agent = 'subred_webscraper'

        # Create a directory to save the downloaded GIFs
        self.save_directory = f'web_scraper/reddit/{self.subreddit}/{self.tag}/reddit_videos'

    def _scrape(self,
                start_time="2022-01-01",
                end_time="2022-12-30",):

        # Convert times given to Unix timestamps (in seconds)
        self.start_time = convert_to_unix_timestamps(start_time)
        self.end_time = convert_to_unix_timestamps(end_time)

        # Create a directory to save the downloaded GIFs
        save_directory = 'reddit_videos'
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        # Create a Reddit API instance
        reddit = praw.Reddit(client_id=self.client_id,
                             client_secret=self.client_secret,
                             user_agent=self.user_agent,
                             limit=1000)

        # Get the subreddit instance
        sub = reddit.subreddit(self.subreddit)

        # Fetch the posts with the specified tag
        posts = sub.search(f'title:{self.tag}', sort='new', time_filter='all')
        # print(f"Got {len(posts._listing.children)} total posts to download from")
        # Iterate over the posts and download the GIFs
        for i, post in enumerate(posts):
            url = post.url
            if 'v.redd.it' in url:
                video_url = url + '/DASH_720.mp4'
                video_filename =  video_url.split('/')[-1].replace("720", str(i))
                video_path = os.path.join(save_directory, video_filename)
                
                # Download the video
                video_response = requests.get(video_url)
                with open(video_path, 'wb') as f:
                    f.write(video_response.content)
                print(f'Downloaded: {video_filename}')

        print('Scraping completed!')

if __name__ == '__main__':
    scraper = subreddit_scraper()

    scraper._scrape()