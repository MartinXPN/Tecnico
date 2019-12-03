export interface SeaGlaciersData {
    year: number;
    level: number;
    mass: number;
}

export interface GdpTemperatureMeatGhgData {
    country: string;
    year: number;
    gdp: number;
    meat_consumption: number;
    temperature: number;
    ghg_emission: number;
}

export interface TemperatureData {
    dt: number;
    Latitude: number;
    Longitude: number;
    AverageTemperature: number;
}
