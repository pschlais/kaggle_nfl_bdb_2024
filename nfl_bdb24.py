import pandas as pd
import numpy as np


# --------- CONSTANTS --------------------------- ####
BDB24_BAD_PLAYS = [
    (2022091102, 4102),  #Play is cut off (not completed)
    (2022110609, 4313),  #Lateral - cut off (tackle not captured)
    (2022102000, 2710),  #Bad data
    (2022110608, 2679),  #Bad data
    (2022100600, 2633),  #Bad data
    (2022091107, 2530),  #Harrison does not appear to make the tackle
    (2022100905, 1254),  #Misidentified tackler
    (2022101600, 376),  #Misidentified tackler
    (2022100213, 1806),  #Doesn't look like Reid made the tackle
    (2022092512, 1299)   #Doesn't look like Barrett made the tackle
]

# ---------- INTERNAL FUNCTIONS ----------------- ####
def _get_tackle_window_data(gap_hist_df: pd.DataFrame) -> pd.Series:
    """
    Outputs the contact and tackle frameIds.

    :param pd.DataFrame gap_hist_df: DataFrame output by _get_tackle_components_vs_time()

    :return pd.Series: [tackleFrameId: int, contactFrameId: int]
    """
    # get tackle event frameId
    tackle_data = gap_hist_df.loc[gap_hist_df.event=='tackle'].iloc[0]
    tackle_frame_id = tackle_data.frameId
    tackle_gap = tackle_data.gap
    # create a boolean index for the 3 seconds of frames before the tackle
    idx_contact_window = (gap_hist_df.frameId >= max(1, tackle_frame_id - 30)) & (gap_hist_df.frameId < tackle_frame_id)

    # check for first_contact event inside contact window
    if len(gap_hist_df.loc[idx_contact_window & (gap_hist_df.event=='first_contact'), :]) > 0:
        # get the first_contact frameId
        first_contact_ser = gap_hist_df.loc[gap_hist_df.event=='first_contact', :].iloc[0]
        gap_at_first_contact = first_contact_ser.gap

        if gap_at_first_contact < 3: # most likely the tackler is the contact defender (provides some error leeway in tracking data positions)
            contact_frame_id = first_contact_ser.frameId
            
        else: # calculate the contact point (same logic as "else" block below)
            # get minimum gap within 3 seconds of the tackle
            min_gap = gap_hist_df.gap[idx_contact_window].min()
            gap_for_contact = max(1.8, min_gap)  # if a wrap-up tackle, find the "threshold" frame (1.8). If a trip, allow for sensor error and find where they are closest and say that is contact frame
            # get earliest frame where gap is below contact threshold
            contact_data = gap_hist_df.loc[idx_contact_window & (gap_hist_df.gap <= (gap_for_contact + 1e-5))].sort_values('frameId', ascending=True).iloc[0]
            contact_frame_id = contact_data.frameId

    else: # calculate the contact point
        # get minimum gap within 3 seconds of the tackle
        min_gap = gap_hist_df.gap[idx_contact_window].min()
        gap_for_contact = max(1.8, min_gap)  # if a wrap-up tackle, find the "threshold" frame (1.8). If a trip, allow for sensor error and find where they are closest and say that is contact frame
        # get earliest frame where gap is below contact threshold
        contact_data = gap_hist_df.loc[idx_contact_window & (gap_hist_df.gap <= (gap_for_contact + 1e-5))].sort_values('frameId', ascending=True).iloc[0]
        contact_frame_id = contact_data.frameId

    # return pd.Series([tackle_frame_id, contact_frame_id, tackle_gap], index=['tackleFrameId', 'contactFrameId', 'tackleGap'], dtype=object)
    return pd.Series([tackle_frame_id, contact_frame_id], index=['tackleFrameId', 'contactFrameId'], dtype=object)

# -------------- FUNCTIONS ------------------------ ####


# helper function for get_tackle_metrics() below
def _get_tackle_components_vs_time(track_df: pd.DataFrame, play_df: pd.DataFrame, tackle_df: pd.DataFrame, player_df: pd.DataFrame) -> pd.DataFrame:
    """
    Outputs the contact and tackle frameIds. Must be used on a single play or pd.Groupby(gameId, playId).

    :param pd.DataFrame track_df: baseline tracking data, MUST BE TRANSFORMED INTO playDirection = right (use nflutil.transform_tracking_data())
    :param pd.DataFrame play_df: data from plays.csv
    :param pd.DataFrame tackle_df: data from tackles.csv
    :param pd.DataFrame player_df: data from players.csv

    :return pd.DataFrame: outputs metrics by frameId. "_t" suffix is for the tackler. No suffix is for the ballcarrier.
    """
    # throw an error if the play is not transformed into the common frame (playDirection = right)
    game_id = track_df.gameId.iloc[0]
    play_id = track_df.playId.iloc[0]
    play_direction = track_df.playDirection.iloc[0]
    if play_direction != 'right':
        raise ValueError(f'playDirection must be "right". Actual playDirection: "{play_direction}". (gameId: {game_id}, playId: {play_id})')

    # get the frame data for the ballcarrier (defined in play_df)
    ballcarrier_df = (
       track_df
        .merge(play_df[['gameId','playId','ballCarrierId']],
            how='inner',
            left_on=['gameId','playId','nflId'],
            right_on=['gameId','playId','ballCarrierId'])
        .drop(columns=['ballCarrierId'])
        # add player weight to the dataset
        .merge(player_df[['nflId', 'weight']],
               on='nflId',
               how='left')
        # set index to unique time stamp for simpler join
        .set_index(['gameId','playId','frameId'])
    )

    # get the frame data for the eventual tackler (tackle = 1 in tackle_df)
    tackler_df = (
        track_df
        .merge(tackle_df.loc[tackle_df.tackle==1, ['gameId','playId','nflId']],
            how='inner',
            on=['gameId','playId','nflId'])
        # add player weight to the dataset
        .merge(player_df[['nflId', 'weight']],
               on='nflId',
               how='left')        
        # set index to unique time stamp for simpler join
        .set_index(['gameId','playId','frameId'])
    )

    # generate the output dataframe applied to the groupby - in this case min distance between
    # ballcarrier and tackler at that point in the play, and the event corresponding to that (if there was one)
    outcome_df = (
        ballcarrier_df
        .merge(tackler_df[['x','y','s','dir','dis','weight']],
            how='inner',
            on=['gameId','playId','frameId'],
            suffixes=(None, '_t'))
        .assign(gap=lambda df: np.sqrt((df['x'] - df['x_t'])**2 + (df['y'] - df['y_t'])**2),
                s_downfield=lambda df: df['s'] * np.sin(np.deg2rad(df['dir'])),
                s_downfield_t=lambda df: df['s_t'] * np.sin(np.deg2rad(df['dir_t']))
                )
        .reset_index()
        [['frameId','event','gap','s','s_downfield','weight','s_downfield_t','weight_t','dis_t','x_t','y_t']]
    )

    return outcome_df


# function version to use on groupby.apply of (gameId, playId)
def prep_get_tackle_metrics(track_df: pd.DataFrame, play_df: pd.DataFrame, tackle_df: pd.DataFrame, player_df: pd.DataFrame) -> pd.Series:
    """
    PREPROCESSING - Outputs tackle metrics for a given play to be used for generating features (gameId-playId combo). Must be used on a single play or pd.Groupby(gameId, playId).

    :param pd.DataFrame track_df: baseline tracking data, MUST BE TRANSFORMED INTO playDirection = right (use nflutil.transform_tracking_data())
    :param pd.DataFrame play_df: data from plays.csv
    :param pd.DataFrame tackle_df: data from tackles.csv
    :param pd.DataFrame player_df: data from players.csv

    :return pd.Series: output metrics for the play.
    """
    # get component data by frame
    comp_df = _get_tackle_components_vs_time(track_df, play_df, tackle_df, player_df).set_index('frameId')

    # get the contact frames
    tackle_frame_id_ser = _get_tackle_window_data(comp_df.reset_index())
    contact_frame_id = tackle_frame_id_ser.contactFrameId
    tackle_frame_id = tackle_frame_id_ser.tackleFrameId
    tackle_window_frames = tackle_frame_id - contact_frame_id  # number of frames between contact frame and tackle frame - process of the tackle

    #### Generate metrics

    # - Vision: Efficiency of defender path - #
    path_start_frame_id = max(1, contact_frame_id - 30)  # start 3 seconds before contact for pathing - can't go before frame 1
    idx_mask_actual = (comp_df.index > path_start_frame_id) & (comp_df.index <= contact_frame_id)
    d_actual = comp_df.loc[idx_mask_actual, 'dis_t'].sum()
    d_ideal = np.sqrt((comp_df.x_t.loc[contact_frame_id] - comp_df.x_t.loc[path_start_frame_id])**2 
                      + (comp_df.y_t.loc[contact_frame_id] - comp_df.y_t.loc[path_start_frame_id])**2)
    
    if d_actual == 0: # no lead-up, so don't give an efficiency rating
        d_eff = np.nan
    else:
        d_eff = d_ideal / d_actual

    # - Drive through tackle: Momentum change due to interaction, relative to neutral interaction - #
    w_carrier = comp_df.weight.iloc[0]
    w_tackler = comp_df.weight_t.iloc[0]
    s_downfield_neutral = comp_df.s_downfield.loc[contact_frame_id] * w_carrier / (w_carrier + w_tackler) # inelastic collision where defender "hugs" and does not impart a force
    s_downfield_tackle_m1 = comp_df.s_downfield.loc[tackle_frame_id - 1]  # get downfield velocity the frame before the tackle
    s_downfield_delta = s_downfield_tackle_m1 - s_downfield_neutral  # >0: going downfield faster than neutral interaction (bad for defense)
    s_contact = comp_df.s.loc[contact_frame_id]  # speed at contact (magnitude)

    # - Wrap up: Gap at tackle event - #
    gap_tackle = comp_df.gap.loc[tackle_frame_id]

    # send back metrics
    return pd.Series([contact_frame_id, 
                      tackle_frame_id,
                      tackle_window_frames,
                      d_actual, 
                      d_ideal, 
                      d_eff, 
                      gap_tackle,
                      w_carrier,
                      w_tackler,
                      s_downfield_delta,
                      s_contact
                      ], 
                      index=['contactFrameId',
                             'tackleFrameId',
                             'frames',
                             'd_actual',
                             'd_ideal',
                             'd_eff', 
                             'gap_tackle',
                             'w_carrier',
                             'w_tackler',
                             's_downfield_delta',
                             's_contact'
                             ],
                     dtype=object
                    )

def util_play_contains_tackle(track_df: pd.DataFrame) -> bool:
    # validation - check to make sure only one play is passed into the function
    game_list = track_df.gameId.unique().tolist()
    play_list = track_df.playId.unique().tolist()
    if len(game_list) > 1 or len(play_list) > 1:
        raise ValueError(f'track_df passed in contained more than one play. gameId: {game_list}, playId: {play_list}')
    # return True if play ends out of bounds
    return len(track_df.loc[track_df.event=='tackle', ['event']]) > 0

def util_play_contains_qb_slide(track_df: pd.DataFrame) -> bool:
    # validation - check to make sure only one play is passed into the function
    game_list = track_df.gameId.unique().tolist()
    play_list = track_df.playId.unique().tolist()
    if len(game_list) > 1 or len(play_list) > 1:
        raise ValueError(f'track_df passed in contained more than one play. gameId: {game_list}, playId: {play_list}')
    # return True if play ends out of bounds
    return len(track_df.loc[track_df.event=='qb_slide', ['event']]) > 0