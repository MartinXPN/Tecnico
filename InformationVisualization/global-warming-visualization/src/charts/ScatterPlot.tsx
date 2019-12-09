import React, {Component} from "react";
import * as d3 from "d3";
import {GdpTemperatureMeatGhgData} from "../entities";
import * as _ from "lodash";
import Tooltip from "../tooltip/Tooltip";

export interface Props {
    width: number | string;
    height: number | string;
    data: d3.DSVParsedArray<GdpTemperatureMeatGhgData>;

    yearStart: number;
    yearEnd: number;
    startColor: string;
    endColor: string;

    selectedCountries: Set<string>;
    selectCountry: (country: string) => void;
    hoverCountry: (country: string | undefined) => void;
    currentHoveredCountry: string | undefined;
}

interface State {
    countriesDisplayed: Set<string>;
}

export default class ScatterPlot extends Component<Props, State> {
    private readonly countryToData: Map<string, Map<number, GdpTemperatureMeatGhgData>>;
    // @ts-ignore
    protected ref: SVGSVGElement;
    // @ts-ignore
    protected xScale: d3.ScaleLinear<number, number>;
    // @ts-ignore
    protected xAxis: d3.Selection<SVGGElement, unknown, null, undefined>;
    // @ts-ignore
    protected yScale: d3.ScaleLinear<number, number>;
    // @ts-ignore
    protected yAxis: d3.Selection<SVGGElement, unknown, null, undefined>;
    protected tooltip: Tooltip;
    protected xLabel = 'GHG Emissions';
    protected yLabel = 'Temperature ℃';
    protected title = 'Temperature and GHG emissions';
    protected static OPACITIES = {DISABLED: 0.1, ENABLED: 0.7, HIGHLIGHTED: 1};

    state = {
        countriesDisplayed: new Set<string>(),
    };

    constructor(props: Props) {
        super(props);
        this.tooltip = new Tooltip({});

        this.countryToData = new Map();
        props.data.forEach(d => {
            if (!this.countryToData.has(d.country))
                this.countryToData.set(d.country, new Map());
            // @ts-ignore
            this.countryToData.get(d.country).set(d.year, d);
        });

        console.log(this.countryToData);
    }

    getX = (d: GdpTemperatureMeatGhgData) => d.ghg_emission;
    getY = (d: GdpTemperatureMeatGhgData) => d.temperature;
    addCountry = (svg: d3.Selection<SVGSVGElement, unknown, null, undefined>, country: string, title: string) => {
        svg.append(`circle`).attr('title', title);
    };
    removeCountry = (svg: d3.Selection<SVGSVGElement, unknown, null, undefined>, country: string, title: string) => {
        svg.select(`circle[title='${title}']`).remove();
    };


    componentDidMount(): void {
        const svg = d3.select(this.ref);
        const rect = this.ref.getBoundingClientRect();

        const w = rect.width;
        const h = rect.height;
        const padding = 50;

        svg.append("text")
            .attr("transform", "translate(" + ((w - padding) / 2) + " ," + (h / 6) + ")")
            .style("text-anchor", "middle")
            .text(this.title)
            .attr('font-size', '15px');

        this.xAxis = svg.append("g")
            .attr("class", "axis")
            .attr("transform", "translate(0," + (h - padding) + ")");

        this.yAxis = svg.append("g")
            .attr("class", "axis")
            .attr("transform", "translate(" + padding + ", 0)");


        //x label
        svg.append("text")
            .attr("transform", "translate(" + ((w - padding) / 2) + " ," + (h - 15) + ")")
            .style("text-anchor", "middle")
            .text(this.xLabel)
            .attr('font-size', '12px');

        //y label
        svg.append("text")
            .attr("transform", "rotate(-90)")
            .attr("y", 10)
            .attr("x", -h / 2)
            .attr("dy", "1em")
            .style("text-anchor", "middle")
            .text(this.yLabel)
            .attr('font-size', '12px');

        this.componentDidUpdate(this.props, this.state);
    }

    handleCountryYear = (svg: d3.Selection<SVGSVGElement, unknown, null, undefined>,
                         dataPoint: GdpTemperatureMeatGhgData | undefined,
                         country: string, identifier: string,
                         color: string, h: number) => {
        if (!dataPoint) {
            svg.select(`circle[title='${identifier}-${country}']`).attr('visibility', 'hidden');
            return;
        }

        const hoverFactor = this.props.currentHoveredCountry === country ? 1.5 : 1;
        svg.select(`circle[title='${identifier}-${dataPoint.country}']`)
            .on("mouseover", () => {
                this.props.hoverCountry(country);
                this.tooltip.show(`<div style="text-align: center"><strong>${dataPoint.country}</strong></div> - ${Math.round(dataPoint.ghg_emission / 100000) / 10 + 'M'} greenhouse gas emissions<div> - ${dataPoint.temperature}℃ average yearly temperature</div>`);
            })
            .on("mousemove", () => this.tooltip.move(d3.event.pageY - 10, d3.event.pageX + 10))
            .on("mouseout", () => {
                this.props.hoverCountry(undefined);
                this.tooltip.hide();
            })
            .on("click", () => this.props.selectCountry(country))
            .transition().duration(150)
            .attr('cx', this.xScale(this.getX(dataPoint)))
            .attr('cy', h - this.yScale(this.getY(dataPoint)))
            .attr('r', hoverFactor * 4)
            .attr("fill", color)
            .attr('visibility', 'visible')
            .attr('opacity', this.props.currentHoveredCountry === country ? ScatterPlot.OPACITIES.HIGHLIGHTED : ScatterPlot.OPACITIES.ENABLED);
    };

    componentDidUpdate(prevProps: Readonly<Props>, prevState: Readonly<{}>, snapshot?: any): void {
        const svg = d3.select(this.ref);
        const rect = this.ref.getBoundingClientRect();

        const w = rect.width;
        const h = rect.height;
        const padding = 50;

        // remove old countries
        this.state.countriesDisplayed.forEach(country => {
            if (!this.props.selectedCountries.has(country)) {
                this.removeCountry(svg, country, `yearStart-${country}`);
                this.removeCountry(svg, country, `yearEnd-${country}`);
            }
        });

        const filter = (d: any) => this.props.selectedCountries.has(d.country);
        // && (d.year === this.props.yearStart || d.year === this.props.yearEnd);
        let maxX = d3.max(this.props.data, d => filter(d) ? this.getX(d) : 0);
        let minX = d3.min(this.props.data, d => filter(d) ? this.getX(d) : Infinity);
        let maxY = d3.max(this.props.data, d => filter(d) ? this.getY(d) : 0);
        let minY = d3.min(this.props.data, d => filter(d) ? this.getY(d) : Infinity);
        if (maxX === undefined || minX === undefined || minY === undefined || maxY === undefined)
            return;

        const rangeX = maxX - minX === 0 ? maxX : maxX - minX;
        const rangeY = maxY - minY === 0 ? maxY : maxY - minY;

        minX -= 0.1 * rangeX;
        maxX += 0.1 * rangeX;
        minY -= 0.1 * rangeY;
        maxY += 0.1 * rangeY;
        this.xScale = d3.scaleLinear()
            .domain([minX, maxX])
            .range([padding, w - 2 * padding]);

        this.yScale = d3.scaleLinear()
            .domain([minY, maxY])
            .range([h - padding, padding]);

        const xAxis = d3.axisBottom(this.xScale)
            .ticks(5)
            // @ts-ignore
            .tickFormat((val: number, id: number) => {
                if (!maxX) return id;
                if (maxX > 5 * Math.pow(10, 6)) return '' + Math.round(val / 1000000) + 'M';
                if (maxX > 5 * Math.pow(10, 3)) return '' + Math.round(val / 1000) + 'K';
                return '' + val;
            });
        const yAxis = d3.axisLeft(this.yScale).ticks(5);

        this.xAxis.call(xAxis);
        this.yAxis.call(yAxis);


        // add new countries and display the data
        this.props.selectedCountries.forEach(country => {
            if (!this.state.countriesDisplayed.has(country)) {
                this.addCountry(svg, country, `yearStart-${country}`);
                this.addCountry(svg, country, `yearEnd-${country}`);
            }

            const countryData = this.countryToData.get(country);
            if(!countryData)
                return;

            const start = countryData.get(this.props.yearStart);
            const end = countryData.get(this.props.yearEnd);

            this.handleCountryYear(svg, start, country, 'yearStart', this.props.startColor, h);
            this.handleCountryYear(svg, end, country, 'yearEnd', this.props.endColor, h);
        });

        if (!_.isEqual(this.props.selectedCountries, this.state.countriesDisplayed))
            this.setState({countriesDisplayed: this.props.selectedCountries});
    }

    render(): React.ReactElement {
        return (
            <svg
                ref={(ref: SVGSVGElement) => this.ref = ref}
                width={this.props.width}
                height={this.props.height}/>
        );
    }
}
