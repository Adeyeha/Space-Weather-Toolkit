"""
 * NRT-CME-Flare-Processor, a project at the Data Mining Lab
 * (http://dmlab.cs.gsu.edu/) of Georgia State University (http://www.gsu.edu/).
 *
 * Copyright (C) 2020 Georgia State University
 *
 * This program is free software: you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free Software
 * Foundation version 3.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * this program. If not, see <http://www.gnu.org/licenses/>.
"""

import requests
import traceback
import pandas as pd

from typing import List
from logging import Logger
from pandas import DataFrame
from datetime import datetime,timedelta
from collections import OrderedDict
from . import ReportType,ReportCat,VERSION
    
class CMEDownloader:
    """
    This is a base abstract class for download CME reports.
    """

    def __init__(self, logger: Logger):
        """
        Constructor
        :param Logger:
        """
        self._latest_report_url = "https://www.sidc.be/cactus/out/cmecat.txt"
        self._logger = logger
        self._available_reports = [str(member.name) for member in ReportType]
        self._available_categories = [str(member.name) for member in ReportCat]
        self._ReportTypeConfig = dict((i.name,i.value) for i in ReportType)
        self._ReportCatConfig = dict((i.name,i.value) for i in ReportCat)
        self._dateString = r"%d-%m-%Y"

    def _validate_parameters(self,report:str,category:str,start_time:datetime,end_time:datetime):
        
        if report not in self._available_reports:
            raise ValueError(f"'{report}' is not a valid report. Should be one of {','.join(self._available_reports)}")

        if category not in self._available_categories:
            raise ValueError(f"'{category}' is not a valid category. Should be one of {','.join(self._available_categories)}")
        
        if end_time < start_time:
            raise ValueError(f"start_time cannot be after end_time")
            
        return True

    def _merge_with_latest_report(self,dates:list,records:pd.DataFrame,report:str):
        """
        Merges the reports from the catalog for processed CME data with the lastest CME report for complete report rendering
        
        :param dates: list of all dates within date range
        :param records: pandas dataframe for reports from catalog
        :param report: CME or FLOW reports
        :return: Merged report
        """
        resp = requests.get(url=self._latest_report_url)
        if resp.status_code == 200:
            data = resp.text
            dt = self._format_report(data,report,dates)
            
            if dt.shape[0] > 0:
                
                # Merge reports from catalog with latest report
                for date in list(set(dt.t.dt.strftime(self._dateString))):
                    latest_date = records[records.t.dt.strftime(self._dateString) == date].t.max()
                    if  not pd.isnull(latest_date):
                        records = pd.concat([records,dt[(dt.t.dt.strftime(self._dateString) == date) & (dt.t > latest_date)]],axis=0)
                    else:
                        records = pd.concat([records,dt[(dt.t.dt.strftime(self._dateString) == date)]],axis=0)
        
        return records 
        
    def _format_report(self,data:pd.DataFrame ,report:str,dates:list):
        """
        Formats report downloaded from catalog.
        
        :param dates: list of all dates within date range
        :param data: pandas dataframe for reports from catalog
        :param report: CME or FLOW reports
        :return: Merged report
        """
        
        # Extract issue date and phrase date
        issue_date_str = data.split('\n:Product:')[0].split(':Issued: ')[1]
        issue_date = datetime.strptime(issue_date_str, '%a %b %d %H:%M:%S %Y')

        # Extract new data
        new_data = data.split("#")[self._ReportTypeConfig[report]].strip().split("\n")
        new_data = [element.split('|') for element in new_data]
        names = new_data.pop(0)

        # Check if data field is empty and construct dataframe

        if len(new_data) > 0:
            names = ["".join(filter(str.isalpha,name.strip())) for name in names]
            dt = pd.DataFrame(data=new_data,columns=names)
            dt['Issue Date'] = issue_date
            dt['t'] = pd.to_datetime(dt['t'])
            dt = dt[dt.t.dt.strftime(self._dateString).isin(dates)]
            return dt
        
        return None
    

    def get_reports(self,start_time: datetime, end_time: datetime, report: str = "CME", category:str = 'QUICKLOOK') -> List[DataFrame]:

        """
        Downloads the CME or FLOW reports from the Cactus catalog.
        
        :param start_time: The start time of the range.
        :param end_time: The end time of the range.
        :param report: Indicator for report type. Default is CME.
        :param category: Category of report. Default is QUICKLOOK.
        :return: Final CME/FLOW report
        """
        
        # Params validation
        self._validate_parameters(report,category,start_time,end_time)
        
        # Get all parameters to form report_url
        version = VERSION.replace('.', '_')
        dates = OrderedDict(((start_time + timedelta(_)).strftime(self._dateString), None) for _ in range((end_time - start_time).days + 1)).keys()
        date_partitions = OrderedDict(((start_time + timedelta(_)).strftime(r"%m-%Y"), None) for _ in range((end_time - start_time).days + 1)).keys()
        result = pd.DataFrame()
        
        # Loop through month and year pairs
        try:
            
            for date in date_partitions:
                month,year = date.split('-')
                report_url = f"https://www.sidc.be/cactus/catalog/LASCO/{version}/{self._ReportCatConfig[category]}/{year}/{month}/cmecat.txt"

                resp = requests.get(url=report_url)
                if resp.status_code == 200:
                    data = resp.text
                    dt = self._format_report(data,report,dates)
                    if dt is not None:
                        result = pd.concat([result,dt],axis=0)

            # Merge with latest report
            result = self._merge_with_latest_report(dates,result,report)
            
            # Convert datatypes 
            result = result.drop(result.columns[0],axis=1).reset_index(drop=True)
            for x in result.select_dtypes(exclude=['datetime64[ns]']).columns:
                try:
                    result[x] = result[x].astype(int)
                except Exception as e:
                    pass
            
                        
        except Exception as e:
            self._logger.critical("CMEDownloader.get_reports failed with error: %s", str(e))
            self._logger.critical("CMEDownloader.get_reports failed traceback: %s", traceback.format_exc())

        return result
