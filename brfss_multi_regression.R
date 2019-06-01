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





