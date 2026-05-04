import pandas as pd
from supabase import create_client
from data_collection.config import SUPABASE_URL, SUPABASE_KEY

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

weather = pd.DataFrame(supabase.table("match_weather").select("*").execute().data)
coach   = pd.DataFrame(supabase.table("match_coach").select("*").execute().data)
stadium = pd.DataFrame(supabase.table("match_stadium").select("*").execute().data)
referee = pd.DataFrame(supabase.table("match_referee").select("*").execute().data)

df = weather.merge(coach, on="match_id", how="left") \
            .merge(stadium, on="match_id", how="left") \
            .merge(referee, on="match_id", how="left")

df.to_csv("qsport_dataset.csv", index=False)
print(f"{len(df)} matchs exportés vers qsport_dataset.csv")
