library(haven)
library(tidyverse)
library(dplyr)
library(ggplot2)


##Store data file to variable survey17
setwd("Documents/OSU/courses/Su_19/project/brfss")
getwd()
survey11 <- read_xpt("LLCP2011.XPT")
survey12 <- read_xpt("LLCP2012.XPT")
survey13 <- read_xpt("LLCP2013.XPT ")
survey14 <- read_xpt("LLCP2014.XPT ")
survey15 <- read_xpt("LLCP2015.XPT ")
survey16 <- read_xpt("LLCP2016.XPT ")
survey17 <- read_xpt("LLCP2017.XPT ")

tail(survey11)
tail(survey17)

variables57 = c("_STATE", "IMONTH", "IYEAR", "DIABETE3", "DIABAGE2", "_RFHLTH", "HLTHPLN1", "MEDCOST", "CHECKUP1", "_AGEG5YR", "SEX", "CHILDREN", "_EDUCAG", "MARITAL", "RENTHOM1", "EMPLOY1", "INCOME2", "_BMI5", "_SMOKER3", "_RFDRHV5")
variables34 = c("_STATE", "IMONTH", "IYEAR", "DIABETE3", "DIABAGE2", "_RFHLTH", "HLTHPLN1", "MEDCOST", "CHECKUP1", "_AGEG5YR", "SEX", "CHILDREN", "_EDUCAG", "MARITAL", "RENTHOM1", "EMPLOY1", "INCOME2", "_BMI5", "_SMOKER3", "_RFDRHV4")
variables12 = c("_STATE", "IMONTH", "IYEAR", "DIABETE3", "DIABAGE2", "_RFHLTH", "HLTHPLN1", "MEDCOST", "CHECKUP1", "_AGEG5YR", "SEX", "CHILDREN", "_EDUCAG", "MARITAL", "RENTHOM1", "EMPLOY", "INCOME2", "_BMI5", "_SMOKER3", "_RFDRHV4")

rsurvey11 <- select(survey11, variables12)
rsurvey12 <- select(survey12, variables12)
rsurvey13 <- select(survey13, variables34)
rsurvey14 <- select(survey14, variables34)
rsurvey15 <- select(survey15, variables57)
rsurvey16 <- select(survey16, variables57)
rsurvey17 <- select(survey17, variables57)

rsurvey14 <- rename(rsurvey14, "_RFDRHV5" = "_RFDRHV4")
rsurvey13 <- rename(rsurvey13, "_RFDRHV5" = "_RFDRHV4")
rsurvey12 <- rename(rsurvey12, "_RFDRHV5" = "_RFDRHV4", "EMPLOY1" = "EMPLOY")
rsurvey11 <- rename(rsurvey11, "_RFDRHV5" = "_RFDRHV4", "EMPLOY1" = "EMPLOY")










