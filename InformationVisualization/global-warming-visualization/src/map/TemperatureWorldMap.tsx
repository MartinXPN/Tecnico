import React, {Component} from "react";
import * as d3 from 'd3';
// @ts-ignore
import * as simpleheat from 'simpleheat';
import {CountryTemperatureData, TemperatureData} from "../entities";
import Tooltip from "../tooltip/Tooltip";

interface Props {
    width: number | string;
    height: number | string;
    yearStart: number;
    yearEnd: number;
    hoverCountry: (country: string | undefined) => void;
    hoveredCountry: string | undefined;
    selectedCountries: Set<string>;
    addCountry: (country: string) => void;
    removeCountry: (country: string) => void;
    data: d3.DSVParsedArray<CountryTemperatureData>;
}

interface State {
    currentlyHoveredCountry: string | undefined;
    temperatureData: d3.DSVParsedArray<TemperatureData>;
    countryData: Map<string, Map<number, number>>;
}

export default class TemperatureWorldMap extends Component<Props, State> {
    private static HOVER_OPACITY = 0.8;
    private static NORMAL_OPACITY = 0.5;
    private static COLOR = 'rgba(0,0,0,0.5)';
    private static HOVER_STROKE = 2;
    private static NORMAL_STROKE = 0.3;

    // @ts-ignore
    private ref: SVGSVGElement;
    // @ts-ignore
    private map: d3.Selection<SVGPathElement, unknown, SVGGElement, unknown>;
    // @ts-ignore
    private heat: any;
    // @ts-ignore
    private projection: d3.GeoProjection;
    private tooltip: Tooltip = new Tooltip({});

    // @ts-ignore
    state = {
        currentlyHoveredCountry: undefined,
        temperatureData: [],
        countryData: new Map(),
    };


    drawMap = () => {
        const rect = this.ref.getBoundingClientRect();
        const w = rect.width;
        const h = rect.height;

        this.projection = d3.geoMercator()
            .scale(155)
            .translate([w / 2, h / 1.7]);
        const path = d3.geoPath().projection(this.projection);

        // @ts-ignore
        this.map.attr('d', path)
            .style('fill', TemperatureWorldMap.COLOR)
            .style('stroke', 'white')
            .style('opacity', (d: any) => d.properties.name === this.state.currentlyHoveredCountry ? TemperatureWorldMap.HOVER_OPACITY : TemperatureWorldMap.NORMAL_OPACITY)
            .style('stroke-width', (d: any) => d.properties.name === this.state.currentlyHoveredCountry ? TemperatureWorldMap.HOVER_STROKE : TemperatureWorldMap.NORMAL_STROKE);

    };

    drawHeatMap = () => {

        const start: Map<string, number> = new Map();
        const end: Map<string, number> = new Map();
        this.state.temperatureData.forEach((record: TemperatureData) => {
            if (record.dt === this.props.yearStart) start.set(record.Longitude + '#' + record.Latitude, record.AverageTemperature);
            if (record.dt === this.props.yearEnd)   end.set(record.Longitude + '#' + record.Latitude, record.AverageTemperature);
        });

        const temperatureDifference = [];
        // @ts-ignore
        for (const cord of start.keys()) {
            if (end.has(cord)) {
                const [lng, lat] = cord.split('#');
                const lnglat = this.projection([+lng, +lat]);
                // @ts-ignore
                temperatureDifference.push([lnglat[0], lnglat[1], end.get(cord) - start.get(cord)]);
            }
        }

        this.heat.data(temperatureDifference);
        this.heat.radius(4, 4);
        this.heat.max(d3.max(temperatureDifference, d => d[2]));
        this.heat.draw(0.05);
    };


    componentDidUpdate(prevProps: Readonly<Props>, prevState: Readonly<{}>, snapshot?: any): void {
        const map = d3.select(this.ref);

        if (this.state.currentlyHoveredCountry !== this.props.hoveredCountry) {
            if (this.state.currentlyHoveredCountry) {
                map.select(`[title='${this.state.currentlyHoveredCountry}']`)
                    .style('opacity', TemperatureWorldMap.NORMAL_OPACITY)
                    .style('stroke-width', TemperatureWorldMap.NORMAL_STROKE);
            }

            if (this.props.hoveredCountry) {
                map.select(`[title='${this.props.hoveredCountry}']`)
                    .style('opacity', TemperatureWorldMap.HOVER_OPACITY)
                    .style('stroke-width', TemperatureWorldMap.HOVER_STROKE);
            }

            this.setState({currentlyHoveredCountry: this.props.hoveredCountry});
        }

        /// years were changed => need to render the whole map colors from scratch
        if (this.props.yearStart !== prevProps.yearStart || this.props.yearEnd !== prevProps.yearEnd) {
            this.drawHeatMap();
        }
    }

    componentDidMount() {
        const countryData = new Map<string, Map<number, number>>();
        this.props.data.forEach(d => {
            if (!countryData.has(d.country)) countryData.set(d.country, new Map());
            // @ts-ignore
            if (!countryData.get(d.country).has(d.year)) countryData.get(d.country).set(d.year, d.temperature);
        });
        this.setState({countryData: countryData});
        console.log(countryData);


        const rect = this.ref.getBoundingClientRect();

        const w = rect.width;
        const h = rect.height;

        // heatmap
        const div = d3.select('#container');
        const canvasLayer = div.append('canvas').attr('id', 'heatmap').attr('width', w).attr('height', h);
        const canvas = canvasLayer.node();
        this.heat = simpleheat(canvas);

        // map
        d3.select(this.ref).append("text")
            .attr("transform", "translate(" + (w / 2) + " ," + (h - 10) + ")")
            .style("text-anchor", "middle")
            .text('Temperature difference map for cities with records')
            .attr('font-weight', 'bold')
            .attr('font-size', '15px');

        d3.json('./world_countries.json').then(data => {
            this.map = d3.select(this.ref)
                .append("g")
                .attr('class', 'map')
                .selectAll("path")
                .data(data.features)
                .enter()
                .append("path")
                .attr('title', (d: any) => d.properties.name)
                .on('click', (d: any) => {
                    const country = d.properties.name;
                    if (this.props.selectedCountries.has(country))  this.props.removeCountry(country);
                    else                                            this.props.addCountry(country);
                })
                .on("mouseover", (d: any) => {
                    const country = d.properties.name;
                    this.props.hoverCountry(country);

                    let description = ``;
                    let startTemperature = undefined;
                    let endTemperature = undefined;
                    if (this.state.countryData.has(country) && this.state.countryData.get(country).has(this.props.yearStart)) {
                        startTemperature = this.state.countryData.get(country).get(this.props.yearStart);
                        description += `<div>${this.props.yearStart}: temperature was ${startTemperature}℃</div>`;
                    }
                    if (this.state.countryData.has(country) && this.state.countryData.get(country).has(this.props.yearEnd)) {
                        endTemperature = this.state.countryData.get(country).get(this.props.yearEnd);
                        description += `<div>${this.props.yearEnd}: temperature was ${endTemperature}℃</div>`;
                    }
                    if (startTemperature && endTemperature)
                        description += `<div>Temperature change: ${Math.round((endTemperature - startTemperature) * 100) / 100}℃</div>`;

                    this.tooltip.show(`<div style="text-align: center"><strong>${country}</strong>${description}`);
                })
                .on("mousemove", () => this.tooltip.move(d3.event.pageY - 10, d3.event.pageX + 10))
                .on("mouseout", () => {
                    this.props.hoverCountry(undefined);
                    this.tooltip.hide();
                });

            this.drawMap();
        });

        d3.json('./temperatures_by_city.json').then(data => {
            this.setState({temperatureData: data}, () => this.drawHeatMap());
        });
    }

    render(): React.ReactElement {
        return (
            <div style={{width: this.props.width, height: this.props.height}}>
                <div id='container'
                     style={{position: "absolute", left: 0, top: '10%', bottom: '20%', height: '70%', width: '100%'}}/>
                <svg ref={(ref: SVGSVGElement) => this.ref = ref}
                     style={{position: "absolute", left: 0, top: '10%', bottom: '20%'}}
                     width='100%' height='70%'/>
            </div>
        );
    }
}
