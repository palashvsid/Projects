library(RODBC)
library(ggplot2)
library(xtable)

Local <- odbcConnect("Test", uid = "", pwd = "")

country_details <- sqlQuery(Local, "
select * from  project.dbo.country_details;")

energy_access <- sqlQuery(Local, "
select * from  project.dbo.energy_access;")

energy_efficiency <- sqlQuery(Local, "
select * from  project.dbo.energy_efficiency;")

renewable_consumption <- sqlQuery(Local, "
select * from  project.dbo.renewable_consumption;")

renewable_production <- sqlQuery(Local, "
select * from  project.dbo.renewable_production;")

correlation_analysis <- sqlQuery(Local, "
                              select 
*
                              from  
                                 project.dbo.renewable_consumption a 
                              inner join 
                                 project.dbo.energy_access b
                              on 
                                 a.Country_Name = b.Country_Name
                                 and
                                 a.Year=b.Year
                              inner join
  project.dbo.energy_efficiency c
                              on 
                                 a.Country_Name = c.Country_Name
                                 and
                                 a.Year=c.Year
                                 
inner join
  project.dbo.renewable_production e
                              on 
                                 a.Country_Name = e.Country_Name
                                 and
                                 a.Year=e.Year;"
)

correlation_analysis <- correlation_analysis[-c(15,16,20,21,24,25)]

write.csv(as.data.frame(cor(correlation_analysis[-c(1,2,3,5, 7, 8, 9, 10, 12, 14)])), "corr.csv")
