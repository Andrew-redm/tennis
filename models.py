from sqlalchemy import Table, Column, Integer, String, DateTime, Float, Boolean, MetaData
from pydantic import BaseModel, ConfigDict
import datetime

metadata = MetaData()

games_atp = Table('games_atp', metadata,
    Column('ID1_G', Integer),
    Column('ID2_G', Integer),
    Column('ID_T_G', Integer),
    Column('ID_R_G', Integer),
    Column('RESULT_G', String),
    Column('DATE_G', DateTime),
)

games_wta = Table('games_wta', metadata,
    Column('ID1_G', Integer),
    Column('ID2_G', Integer),
    Column('ID_T_G', Integer),
    Column('ID_R_G', Integer),
    Column('RESULT_G', String),
    Column('DATE_G', DateTime),
)

players_atp = Table('players_atp', metadata,
    Column('ID_P', Integer, primary_key=True),
    Column('NAME_P', String),
    Column('DATE_P', DateTime),
    Column('COUNTRY_P', String),
    Column('RANK_P', Integer),
    Column('PROGRESS_P', Integer),
    Column('POINT_P', Integer),
    Column('HARDPOINT_P', Integer),
    Column('HARDTOUR_P', Integer),
    Column('CLAYPOINT_P', Integer),
    Column('CLAYTOUR_P', Integer),
    Column('GRASSPOINT_P', Integer),
    Column('GRASSTOUR_P', Integer),
    Column('CARPETPOINT_P', Integer),
    Column('CARPETTOUR_P', Integer),
    Column('PRIZE_P', String),
    Column('CH_P', Integer),
    Column('DR_P', Integer),
    Column('DP_P', Integer),
    Column('DO_P', Integer),
    Column('IHARDPOINT_P', Integer),
    Column('IHARDTOUR_P', Integer),
    Column('ITF_ID', String),
)

players_wta = Table('players_wta', metadata,
    Column('ID_P', Integer, primary_key=True),
    Column('NAME_P', String),
    Column('DATE_P', DateTime),
    Column('COUNTRY_P', String),
    Column('RANK_P', Integer),
    Column('PROGRESS_P', Integer),
    Column('POINT_P', Integer),
    Column('HARDPOINT_P', Integer),
    Column('HARDTOUR_P', Integer),
    Column('CLAYPOINT_P', Integer),
    Column('CLAYTOUR_P', Integer),
    Column('GRASSPOINT_P', Integer),
    Column('GRASSTOUR_P', Integer),
    Column('CARPETPOINT_P', Integer),
    Column('CARPETTOUR_P', Integer),
    Column('PRIZE_P', String),
    Column('CH_P', Integer),
    Column('DR_P', Integer),
    Column('DP_P', Integer),
    Column('DO_P', Integer),
    Column('IHARDPOINT_P', Integer),
    Column('IHARDTOUR_P', Integer),
    Column('ITF_ID', String),
)


tours_atp = Table('tours_atp', metadata,
    Column('ID_T', Integer, primary_key=True),
    Column('NAME_T', String),
    Column('ID_C_T', Integer),
    Column('DATE_T', DateTime),
    Column('RANK_T', Integer),
    Column('LINK_T', String),
    Column('COUNTRY_T', String),
    Column('PRIZE_T', String),
    Column('RATING_T', Integer),
    Column('URL_T', String),
    Column('LATITUDE_T', Float),
    Column('LONGITUDE_T', Float),
    Column('SITE_T', String),
    Column('RACE_T', String),
    Column('ENTRY_T', String),
    Column('SINGLES_T', Integer),
    Column('DOUBLES_T', Integer),
    Column('TIER_T', String),
    Column('RESERVE_INT_T', Integer),
    Column('RESERVE_CHAR_T', String),
    Column('LIVE_T', Integer),
    Column('RESULT_T', String),
)

tours_wta = Table('tours_wta', metadata,
    Column('ID_T', Integer, primary_key=True),
    Column('NAME_T', String),
    Column('ID_C_T', Integer),
    Column('DATE_T', DateTime),
    Column('RANK_T', Integer),
    Column('LINK_T', String),
    Column('COUNTRY_T', String),
    Column('PRIZE_T', String),
    Column('RATING_T', Integer),
    Column('URL_T', String),
    Column('LATITUDE_T', Float),
    Column('LONGITUDE_T', Float),
    Column('SITE_T', String),
    Column('RACE_T', String),
    Column('ENTRY_T', String),
    Column('SINGLES_T', Integer),
    Column('DOUBLES_T', Integer),
    Column('TIER_T', String),
    Column('RESERVE_INT_T', Integer),
    Column('RESERVE_CHAR_T', String),
    Column('LIVE_T', Integer),
    Column('RESULT_T', String),
)


stat_atp = Table('stat_atp', metadata,
    Column('ID1', Integer),
    Column('ID2', Integer),
    Column('ID_T', Integer),
    Column('ID_R', Integer),
    Column('FS_1', Integer),
    Column('FSOF_1', Integer),
    Column('ACES_1', Integer),
    Column('DF_1', Integer),
    Column('UE_1', Integer),
    Column('W1S_1', Integer),
    Column('W1SOF_1', Integer),
    Column('W2S_1', Integer),
    Column('W2SOF_1', Integer),
    Column('WIS_1', Integer),
    Column('BP_1', Integer),
    Column('BPOF_1', Integer),
    Column('NA_1', Integer),
    Column('NAOF_1', Integer),
    Column('TPW_1', Integer),
    Column('FAST_1', Integer),
    Column('A1S_1', Integer),
    Column('A2S_1', Integer),
    Column('FS_2', Integer),
    Column('FSOF_2', Integer),
    Column('ACES_2', Integer),
    Column('DF_2', Integer),
    Column('UE_2', Integer),
    Column('W1S_2', Integer),
    Column('W1SOF_2', Integer),
    Column('W2S_2', Integer),
    Column('W2SOF_2', Integer),
    Column('WIS_2', Integer),
    Column('BP_2', Integer),
    Column('BPOF_2', Integer),
    Column('NA_2', Integer),
    Column('NAOF_2', Integer),
    Column('TPW_2', Integer),
    Column('FAST_2', Integer),
    Column('A1S_2', Integer),
    Column('A2S_2', Integer),
    Column('RPW_1', Integer),
    Column('RPWOF_1', Integer),
    Column('RPW_2', Integer),
    Column('RPWOF_2', Integer),
    Column('MT', Integer),
)

stat_wta = Table('stat_wta', metadata,
    Column('ID1', Integer),
    Column('ID2', Integer),
    Column('ID_T', Integer),
    Column('ID_R', Integer),
    Column('FS_1', Integer),
    Column('FSOF_1', Integer),
    Column('ACES_1', Integer),
    Column('DF_1', Integer),
    Column('UE_1', Integer),
    Column('W1S_1', Integer),
    Column('W1SOF_1', Integer),
    Column('W2S_1', Integer),
    Column('W2SOF_1', Integer),
    Column('WIS_1', Integer),
    Column('BP_1', Integer),
    Column('BPOF_1', Integer),
    Column('NA_1', Integer),
    Column('NAOF_1', Integer),
    Column('TPW_1', Integer),
    Column('FAST_1', Integer),
    Column('A1S_1', Integer),
    Column('A2S_1', Integer),
    Column('FS_2', Integer),
    Column('FSOF_2', Integer),
    Column('ACES_2', Integer),
    Column('DF_2', Integer),
    Column('UE_2', Integer),
    Column('W1S_2', Integer),
    Column('W1SOF_2', Integer),
    Column('W2S_2', Integer),
    Column('W2SOF_2', Integer),
    Column('WIS_2', Integer),
    Column('BP_2', Integer),
    Column('BPOF_2', Integer),
    Column('NA_2', Integer),
    Column('NAOF_2', Integer),
    Column('TPW_2', Integer),
    Column('FAST_2', Integer),
    Column('A1S_2', Integer),
    Column('A2S_2', Integer),
    Column('RPW_1', Integer),
    Column('RPWOF_1', Integer),
    Column('RPW_2', Integer),
    Column('RPWOF_2', Integer),
    Column('MT', Integer),
)


ratings_atp = Table('ratings_atp', metadata,
    Column('DATE_R', DateTime),
    Column('ID_P_R', Integer),
    Column('POINT_R', Integer),
    Column('POS_R', Integer),
)

ratings_wta = Table('ratings_wta', metadata,
    Column('DATE_R', DateTime),
    Column('ID_P_R', Integer),
    Column('POINT_R', Integer),
    Column('POS_R', Integer),
)

odds_atp = Table('odds_atp', metadata,
    Column('ID_B_O', Integer),
    Column('ID1_O', Integer),
    Column('ID2_O', Integer),
    Column('ID_T_O', Integer),
    Column('ID_R_O', Integer),
    Column('K1', Float),
    Column('K2', Float),
    Column('TOTAL', Float),
    Column('KTM', Float),
    Column('KTB', Float),
    Column('F1', Float),
    Column('F2', Float),
    Column('KF1', Float),
    Column('KF2', Float),
    Column('K20', Float),
    Column('K21', Float),
    Column('K12', Float),
    Column('K02', Float),
    Column('K30', Float),
    Column('K31', Float),
    Column('K32', Float),
    Column('K23', Float),
    Column('K13', Float),
    Column('K03', Float),
)

odds_wta = Table('odds_wta', metadata,
    Column('ID_B_O', Integer),
    Column('ID1_O', Integer),
    Column('ID2_O', Integer),
    Column('ID_T_O', Integer),
    Column('ID_R_O', Integer),
    Column('K1', Float),
    Column('K2', Float),
    Column('TOTAL', Float),
    Column('KTM', Float),
    Column('KTB', Float),
    Column('F1', Float),
    Column('F2', Float),
    Column('KF1', Float),
    Column('KF2', Float),
    Column('K20', Float),
    Column('K21', Float),
    Column('K12', Float),
    Column('K02', Float),
    Column('K30', Float),
    Column('K31', Float),
    Column('K32', Float),
    Column('K23', Float),
    Column('K13', Float),
    Column('K03', Float),
)

today_atp = Table('today_atp', metadata,
    Column('TOUR', String),
    Column('DATE_GAME', DateTime),
    Column('ID1', Integer),
    Column('ID2', Integer),
    Column('ROUND', String),
    Column('DRAW', String),
    Column('RESULT', String),
    Column('COMPLETE', Boolean),
    Column('LIVE', Boolean),
    Column('TIME_GAME', DateTime),
    Column('RESERVE_INT', Integer),
    Column('RESERVE_CHAR', String),
)

today_wta = Table('today_wta', metadata,
    Column('TOUR', String),
    Column('DATE_GAME', DateTime),
    Column('ID1', Integer),
    Column('ID2', Integer),
    Column('ROUND', String),
    Column('DRAW', String),
    Column('RESULT', String),
    Column('COMPLETE', Boolean),
    Column('LIVE', Boolean),
    Column('TIME_GAME', DateTime),
    Column('RESERVE_INT', Integer),
    Column('RESERVE_CHAR', String),
)