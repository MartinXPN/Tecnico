import React, {Component} from 'react';
import './App.css';
import * as d3 from 'd3'
import TimeSlider from "./sliders/TimeSlider";
import SplitPane from 'react-split-pane';
import TemperatureWorldMap from "./map/TemperatureWorldMap";
import ScatterPlot from "./charts/ScatterPlot";
import BubbleChart from "./charts/BubbleChart";
import RadialBarChart from "./charts/RadialBarChart";
import {CountryTemperatureData, SeaGlaciersData, GdpTemperatureMeatGhgData} from "./entities";
import SearchBox from "./search/SearchBox";


interface Props {
}

interface State {
    sea2glaciers: d3.DSVParsedArray<SeaGlaciersData> | undefined;
    data: d3.DSVParsedArray<GdpTemperatureMeatGhgData> | undefined;
    country2temperature: d3.DSVParsedArray<CountryTemperatureData> | undefined;
    countryList: Array<string> | undefined;

    yearStart: number;
    yearEnd: number;
    startColor: string;
    endColor: string;

    selectedCountries: Set<string>;
    hoveredCountry: string | undefined;
}

export default class App extends Component<Props, State> {

    state = {
        sea2glaciers: undefined,
        country2temperature: undefined,
        countryList: undefined,
        data: undefined,
        yearStart: 1980,
        yearEnd: 2013,
        startColor: '#428f37',
        endColor: '#34568f',
        selectedCountries: new Set(['Spain', 'Portugal', 'United States', 'Armenia', 'France']),
        hoveredCountry: undefined,
    };

    componentDidMount(): void {
        d3.json('./sea2glaciers.json').then(data => {
            this.setState({sea2glaciers: data});
            console.log(data);
        });

        d3.json('./gdp2temp2meat2ghg.json').then(data => {
            this.setState({data: data});
            console.log(data);
        });

        d3.json('./country2temperature.json').then(data => {
            this.setState({country2temperature: data});
            let countries = data.map((d: CountryTemperatureData) => d.country);
            countries = Array.from(new Set(countries));
            this.setState({countryList: countries});
            console.log(data);
            console.log(countries);
        })
    }

    updateHoveredCountry = (country: string | undefined) => this.setState({hoveredCountry: country});
    selectCountry = (country: string) => this.setState({selectedCountries: new Set([country])});
    addCountry = (country: string) => this.setState({selectedCountries: new Set([...Array.from(this.state.selectedCountries), country])});
    removeCountry = (country: string) => {
        const selectedCountries = new Set(Array.from(this.state.selectedCountries));
        selectedCountries.delete(country);
        this.setState({selectedCountries: selectedCountries});
    };


    render(): React.ReactElement {
        return (
            <div className="App">
                <SplitPane className="content" split="vertical" minSize='20%' defaultSize='40%' maxSize='50%'
                           allowResize={true}>
                    <div style={{width: '100%', height: '100%'}}>
                        <div className="chart-box" style={{paddingTop: '1.5em', paddingRight: '3em'}}>
                            {this.state.sea2glaciers &&
                            <RadialBarChart
                                width='100%' height='100%'
                                // @ts-ignore
                                data={this.state.sea2glaciers}
                                yearStart={this.state.yearStart}
                                yearEnd={this.state.yearEnd}
                            />}
                        </div>
                        <div className="chart-box">
                            {this.state.data &&
                            <BubbleChart
                                width='100%' height='100%'
                                // @ts-ignore
                                data={this.state.data}
                                yearStart={this.state.yearStart}
                                yearEnd={this.state.yearEnd}
                                startColor={this.state.startColor}
                                endColor={this.state.endColor}
                                selectedCountries={this.state.selectedCountries}
                                selectCountry={this.selectCountry}
                                hoverCountry={this.updateHoveredCountry}
                                currentHoveredCountry={this.state.hoveredCountry}
                            />}
                        </div>
                        <div className="chart-box">
                            {this.state.data &&
                            <ScatterPlot
                                width='100%' height='100%'
                                // @ts-ignore
                                data={this.state.data}
                                yearStart={this.state.yearStart}
                                yearEnd={this.state.yearEnd}
                                startColor={this.state.startColor}
                                endColor={this.state.endColor}
                                selectedCountries={this.state.selectedCountries}
                                selectCountry={this.selectCountry}
                                hoverCountry={this.updateHoveredCountry}
                                currentHoveredCountry={this.state.hoveredCountry}
                            />}
                        </div>
                    </div>

                    <div style={{width: '100%', height: '100%'}}>
                        <div className="Time-slider">
                            <TimeSlider
                                domain={[1950, 2013]}
                                initialValues={[this.state.yearStart, this.state.yearEnd]}
                                colors={[this.state.startColor, this.state.endColor]}
                                updateValues={(newValues: number[]) => {
                                    this.setState({yearStart: newValues[0], yearEnd: newValues[1]})
                                }}/>
                        </div>

                        {this.state.country2temperature &&
                        <TemperatureWorldMap width='100%' height='100%'
                                             yearStart={this.state.yearStart}
                                             yearEnd={this.state.yearEnd}
                                             selectedCountries={this.state.selectedCountries}
                                             addCountry={this.addCountry}
                                             removeCountry={this.removeCountry}
                                             hoverCountry={this.updateHoveredCountry}
                                             hoveredCountry={this.state.hoveredCountry}
                                             // @ts-ignore
                                             data={this.state.country2temperature}/>}
                    </div>
                </SplitPane>
                <img src={"logo.png"} className="logo" alt=""/>
                <div className="search">
                    {this.state.countryList &&
                    <SearchBox
                        // @ts-ignore
                        countries={this.state.countryList}
                        addCountry={this.addCountry}
                        removeCountry={this.removeCountry}
                        selectedCountries={this.state.selectedCountries}
                    />}
                </div>
            </div>
        );
    }
};
