library(haven)
library(tidyverse)
library(dplyr)
library(ggplot2)


##Store data files to variables
setwd("Documents/OSU/courses/Su_19/project/brfss")
getwd()
survey11 <- read_xpt("LLCP2011.XPT")
survey12 <- read_xpt("LLCP2012.XPT")
survey13 <- read_xpt("LLCP2013.XPT ")
survey14 <- read_xpt("LLCP2014.XPT ")
survey15 <- read_xpt("LLCP2015.XPT ")
survey16 <- read_xpt("LLCP2016.XPT ")
survey17 <- read_xpt("LLCP2017.XPT ")

# Check tails of files
tail(survey11)
tail(survey17)




################################################################################################
################################################################################################



# Select the variables we care about. The names of some variables are different in earlier surveys.
variables57 = c("_STATE", "IMONTH", "IYEAR", "DIABETE3", "DIABAGE2", "_RFHLTH", "HLTHPLN1", "MEDCOST", "CHECKUP1", "_AGEG5YR", "_RACEGR3", "SEX", "CHILDREN", "_EDUCAG", "MARITAL", "RENTHOM1", "EMPLOY1", "INCOME2", "_SMOKER3", "_RFDRHV5", "EXERANY2", "_BMI5")
variables34 = c("_STATE", "IMONTH", "IYEAR", "DIABETE3", "DIABAGE2", "_RFHLTH", "HLTHPLN1", "MEDCOST", "CHECKUP1", "_AGEG5YR", "_RACEGR3", "SEX", "CHILDREN", "_EDUCAG", "MARITAL", "RENTHOM1", "EMPLOY1", "INCOME2", "_SMOKER3", "_RFDRHV4", "EXERANY2", "_BMI5")
variables12 = c("_STATE", "IMONTH", "IYEAR", "DIABETE3", "DIABAGE2", "_RFHLTH", "HLTHPLN1", "MEDCOST", "CHECKUP1", "_AGEG5YR", "_RACEGR2", "SEX", "CHILDREN", "_EDUCAG", "MARITAL", "RENTHOM1", "EMPLOY", "INCOME2", "_SMOKER3", "_RFDRHV4", "EXERANY2", "_BMI5")

rsurvey11 <- select(survey11, variables12)
rsurvey12 <- select(survey12, variables12)
rsurvey13 <- select(survey13, variables34)
rsurvey14 <- select(survey14, variables34)
rsurvey15 <- select(survey15, variables57)
rsurvey16 <- select(survey16, variables57)
rsurvey17 <- select(survey17, variables57)

# Correct the name discrepancies.
rsurvey14 <- rename(rsurvey14, "_RFDRHV5" = "_RFDRHV4")
rsurvey13 <- rename(rsurvey13, "_RFDRHV5" = "_RFDRHV4")
rsurvey12 <- rename(rsurvey12, "_RFDRHV5" = "_RFDRHV4", "_RACEGR3" = "_RACEGR2", "EMPLOY1" = "EMPLOY")
rsurvey11 <- rename(rsurvey11, "_RFDRHV5" = "_RFDRHV4", "_RACEGR3" = "_RACEGR2", "EMPLOY1" = "EMPLOY")

#Combine all years into a single data set
data17 <- bind_rows(rsurvey11, rsurvey12, rsurvey13, rsurvey14, rsurvey15, rsurvey16, rsurvey17)


# Convert columns of categorical variables to factors
data17f <- lapply(data17[1:21], as.factor)

## Convert data set to tibble and add BMI column back in
data17f <- as_tibble(data17f)
data17f <- bind_cols(data17f, data17[22])


## Linear Regression
lmod1 <- lm(DIABETE3 ~ IYEAR * ., filter(data17f, state == 1))


#install.packages("olsrr")
library(olsrr)
ols_step_all_possible(lmod2)



