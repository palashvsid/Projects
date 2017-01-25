use project;

/* main.csv and country.csv imported from import wizard*/

drop table dbo.country_details;
select distinct 
Country_name, 
CountryCode, 
case 
	when [Income Group] = 'High income: nonOECD' then 'High income'
	when [Income Group] = 'High income: OECD' then 'High income'
	when [Income Group] = 'Upper middle income' then 'Upper middle income'
	when [Income Group] = 'Lower middle income' then 'Lower middle income'
	end as [Income Group], 
Region 
into dbo.country_details
from 
	dbo.country a
inner join 
	dbo.main b
on b.Country_Code= a.CountryCode
where Country_Name in ('United States', 'China', 'Japan', 'Germany', 'United Kingdom', 'France', 'India', 'Italy', 'Brazil', 'Canada', 'Korea, Rep.', 'Russian Federation', 'Australia', 'Spain', 'Mexico', 'Indonesia', 'Netherlands', 'Turkey', 'Switzerland', 'Saudi Arabia')
;

drop table dbo.energy_long;
select a.Country_name, b.Indicator_Code, Year, Value
into dbo.energy_long
from
	dbo.country_details a
left join
	dbo.main b
on a.Country_Name = b.Country_Name
CROSS APPLY
(
	VALUES
		(1990, F5),
		(1991, F6),
		(1992, F7),
		(1993, F8),
		(1994, F9),
		(1995, F10),
		(1996, F11),
		(1997, F12),
		(1998, F13),
		(1999, F14),
		(2000, F15),
		(2001, F16),
		(2002, F17),
		(2003, F18),
		(2004, F19),
		(2005, F20),
		(2006, F21),
		(2007, F22),
		(2008, F23),
		(2009, F24),
		(2010, F25)
) x (Year, Value)
;

drop table dbo.energy_long_subset;
select Country_name, Indicator_Code, Year, Value
into dbo.energy_long_subset
from
	dbo.energy_long a
where Indicator_Code in (
'1.1_ACCESS.ELECTRICITY.TOT',
'1.2_ACCESS.ELECTRICITY.RURAL',
'1.3_ACCESS.ELECTRICITY.URBAN',
'8.1.2_FINAL.ENERGY.INTENSITY',
'8.1.1_FINAL.ENERGY.CONSUMPTION',
'4.1_SHARE.RE.IN.ELECTRICITY',
'5.1.1_TOTAL.CAPACITY',
'5.1.2_RE.CAPACITY',
'1.1_TOTAL.FINAL.ENERGY.CONSUM',
'2.1.8_SHARE.WASTE',
'2.1.5_SHARE.WIND',
'2.1.9_SHARE.BIOGAS',
'2.1.7_SHARE.GEOTHERMAL',
'2.1.3_SHARE.HYDRO',
'2.1.4_SHARE.BIOFUELS',
'2.1.10_SHARE.MARINE',
'2.1.2_SHARE.MODERNBIO',
'2.1.1_SHARE.TRADBIO',
'2.1_SHARE.TOTAL.RE.IN.TFEC',
'2.1.6_SHARE.SOLAR'
)	
;

drop table dbo.energy_wide;
select 
	a.Country_Name, 
	a.Year,
	max(CASE WHEN Indicator_Code='1.1_ACCESS.ELECTRICITY.TOT' THEN Value END) AS 'ACCESS.ELECTRICITY.TOT',
	max(CASE WHEN Indicator_Code='1.2_ACCESS.ELECTRICITY.RURAL' THEN Value END) AS 'ACCESS.ELECTRICITY.RURAL',
	max(CASE WHEN Indicator_Code='1.3_ACCESS.ELECTRICITY.URBAN' THEN Value END) AS 'ACCESS.ELECTRICITY.URBAN',
	max(CASE WHEN Indicator_Code='8.1.2_FINAL.ENERGY.INTENSITY' THEN Value END) AS 'FINAL.ENERGY.INTENSITY',
	max(CASE WHEN Indicator_Code='8.1.1_FINAL.ENERGY.CONSUMPTION' THEN Value END) AS 'FINAL.ENERGY.CONSUMPTION',
	max(CASE WHEN Indicator_Code='4.1_SHARE.RE.IN.ELECTRICITY' THEN Value END) AS 'SHARE.RE.IN.ELECTRICITY',
	max(CASE WHEN Indicator_Code='5.1.1_TOTAL.CAPACITY' THEN Value END) AS 'TOTAL.CAPACITY',
	max(CASE WHEN Indicator_Code='5.1.2_RE.CAPACITY' THEN Value END) AS 'RE.CAPACITY',
	max(CASE WHEN Indicator_Code='1.1_TOTAL.FINAL.ENERGY.CONSUM' THEN Value END) AS 'TOTAL.FINAL.ENERGY.CONSUM',
	max(CASE WHEN Indicator_Code='2.1.8_SHARE.WASTE' THEN Value END) AS 'SHARE.WASTE',
	max(CASE WHEN Indicator_Code='2.1.5_SHARE.WIND' THEN Value END) AS 'SHARE.WIND',
	max(CASE WHEN Indicator_Code='2.1.9_SHARE.BIOGAS' THEN Value END) AS 'SHARE.BIOGAS',
	max(CASE WHEN Indicator_Code='2.1.7_SHARE.GEOTHERMAL' THEN Value END) AS 'SHARE.GEOTHERMAL',
	max(CASE WHEN Indicator_Code='2.1.3_SHARE.HYDRO' THEN Value END) AS 'SHARE.HYDRO',
	max(CASE WHEN Indicator_Code='2.1.4_SHARE.BIOFUELS' THEN Value END) AS 'SHARE.BIOFUELS',
	max(CASE WHEN Indicator_Code='2.1.10_SHARE.MARINE' THEN Value END) AS 'SHARE.MARINE',
	max(CASE WHEN Indicator_Code='2.1.2_SHARE.MODERNBIO' THEN Value END) AS 'SHARE.MODERNBIO',
	max(CASE WHEN Indicator_Code='2.1.1_SHARE.TRADBIO' THEN Value END) AS 'SHARE.TRADBIO',
	max(CASE WHEN Indicator_Code='2.1_SHARE.TOTAL.RE.IN.TFEC' THEN Value END) AS 'SHARE.TOTAL.RE.IN.TFEC',
	max(CASE WHEN Indicator_Code='2.1.6_SHARE.SOLAR' THEN Value END) AS 'SHARE.SOLAR'
into dbo.energy_wide
from dbo.energy_long_subset a
group by Country_Name, Year
;


select
	Country_Name,
	Year,
	[ACCESS.ELECTRICITY.TOT],
	[ACCESS.ELECTRICITY.RURAL],
	[ACCESS.ELECTRICITY.URBAN]
into
	dbo.energy_access
from
	dbo.energy_wide
where YEAR in (1990, 2000, 2010);


select
	Country_Name,
	Year,
	[FINAL.ENERGY.INTENSITY],
	[FINAL.ENERGY.CONSUMPTION]
into
	dbo.energy_efficiency
from
	dbo.energy_wide
;


select
	Country_Name,
	Year,
	[SHARE.RE.IN.ELECTRICITY],
	[TOTAL.CAPACITY],
	[RE.CAPACITY]
into
	dbo.renewable_production
from
	dbo.energy_wide
;


select
	Country_Name,
	Year,
	[TOTAL.FINAL.ENERGY.CONSUM],
	[SHARE.WASTE],
	[SHARE.WIND],
	[SHARE.BIOGAS],
	[SHARE.GEOTHERMAL],
	[SHARE.HYDRO],
	[SHARE.BIOFUELS],
	[SHARE.MARINE],
	[SHARE.MODERNBIO],
	[SHARE.TRADBIO],
	[SHARE.TOTAL.RE.IN.TFEC],
	[SHARE.SOLAR]
into
	dbo.renewable_consumption
from
	dbo.energy_wide
;

drop table dbo.country;
drop table dbo.energy_long;
drop table dbo.energy_wide;
drop table dbo.energy_long_subset;
drop table dbo.main;


/* Total number of records in each table: */
SELECT 
    TableName = t.NAME,
    RowCounts = p.rows
FROM 
    sys.tables t
INNER JOIN      
    sys.indexes i ON t.OBJECT_ID = i.object_id
INNER JOIN 
    sys.partitions p ON i.object_id = p.OBJECT_ID AND i.index_id = p.index_id
WHERE 
    t.is_ms_shipped = 0
GROUP BY
    t.NAME, p.Rows
ORDER BY 
    t.Name
;

/* Total number of countries in consideration by Region: */
select 
	Region,
	count(Country_name) as 'Number of Countries'
from
	dbo.country_details a
group by Region
order by count(Country_name) desc;
;

/* Distribution of countries by income level: */
select 
	[Income Group],
	count(Country_name) as 'Number of Countries'
from
	dbo.country_details a
group by [Income Group]
order by count(Country_name) desc;
;

/* Average share of renewable energy consumption by type of renewable energy in 1990: */
select Country_Name, 
	[SHARE.TOTAL.RE.IN.TFEC] as 'Renewable Energy Share in Total %',
	[SHARE.SOLAR] as 'Solar Energy Share in Total %',
	[SHARE.MODERNBIO] as 'Modern Biofuels Share in Total %',
	[SHARE.HYDRO] as 'Hydroenergy Share in Total %' 
from 
	dbo.renewable_consumption
where 
	Year=1990;

/* Average share of renewable energy consumption by type of renewable energy in 2010: */
select Country_Name, 
	[SHARE.TOTAL.RE.IN.TFEC] as 'Renewable Energy Share in Total %',
	[SHARE.SOLAR] as 'Solar Energy Share in Total %',
	[SHARE.MODERNBIO] as 'Modern Biofuels Share in Total %',
	[SHARE.HYDRO] as 'Hydroenergy Share in Total %' 
from 
	dbo.renewable_consumption
where 
	Year=2010;

/* Order countries by their total electricity capacity in 2010: */
select 
	Country_Name,
	[TOTAL.CAPACITY] as  'Total Capacity',
	[SHARE.RE.IN.ELECTRICITY] as 'Share of Renewable Energy'
from
	dbo.renewable_production
where
	Year='2010'
order by [TOTAL.CAPACITY] desc
;

/* Maximum final energy intensity for each country: */
select
	Country_Name,
	max([FINAL.ENERGY.INTENSITY]) as 'Maximum Energy Intensity'
from
	dbo.energy_efficiency
group by Country_Name
order by max([FINAL.ENERGY.INTENSITY]) desc;

/* Countries in 2010 with access to electricity less than 100% overall and their access % values: */
select 
	a.Country_Name,
	[Income Group],
	[ACCESS.ELECTRICITY.TOT] as 'Electricity Access %'
from
	dbo.energy_access a
inner join
	dbo.country_details b
on a.Country_Name = b.Country_Name
where Year='2010' and [ACCESS.ELECTRICITY.TOT]<>100;