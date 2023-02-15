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

from enum import Enum

class ReportType(Enum):
    """
    This contains all possible reporttypes that can be downloaded from Cactus.
    """
    CME = -2
    FLOW = -1
    
class ReportCat(Enum):
    """
    This contains all possible reportcats that can be downloaded from Cactus.
    """
    LEVEL_ZERO = ''
    QUICKLOOK = 'qkl'
    
VERSION = "2.5.0"