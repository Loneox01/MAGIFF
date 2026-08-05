import nflreadpy as nfl


players = nfl.load_players()
teams = nfl.load_teams()
stats = nfl.load_player_stats(
    seasons=[2025],
    summary_level="week",
)

print("Players:")
print(players.shape)
print(players.columns)
print(players.head())

print("\nTeams:")
print(teams.shape)
print(teams.columns)
print(teams.head())

print("\nWeekly stats:")
print(stats.shape)
print(stats.columns)
print(stats.head())