import * as d3 from "d3";
import ScatterPlot, {Props} from "./ScatterPlot";
import {GdpTemperatureMeatGhgData} from "../entities";


export default class BubbleChart extends ScatterPlot {

    constructor(props: Props) {
        super(props);

        this.xLabel = 'GDP per-capita';
        this.yLabel = 'Meat consumption per-capita';
        this.title = 'Meat consumption GDP and GHG emissions';
    }

    getX = (d: GdpTemperatureMeatGhgData) => d.gdp;
    getY = (d: GdpTemperatureMeatGhgData) => d.meat_consumption;
    addCountry = (svg: d3.Selection<SVGSVGElement, unknown, null, undefined>, country: string, title: string) => {
        svg.append(`circle`).attr('title', title);
        const flagPath = `/flags/${country.toLowerCase().replace(' ', '-')}.svg`;
        svg.append('svg:image').attr('title', title).attr('xlink:href', flagPath);
    };
    removeCountry = (svg: d3.Selection<SVGSVGElement, unknown, null, undefined>, country: string, title: string) => {
        svg.select(`circle[title='${title}']`).remove();
        svg.select(`image[title='${title}']`).remove();
    };


    handleCountryYear = (svg: d3.Selection<SVGSVGElement, unknown, null, undefined>,
                         dataPoint: (GdpTemperatureMeatGhgData | undefined),
                         country: string, identifier: string,
                         color: string, h: number) => {
        if (!dataPoint) {
            svg.select(`circle[title='${identifier}-${country}']`).attr('visibility', 'hidden');
            svg.select(`image[title='${identifier}-${country}']`).attr('visibility', 'hidden');
            return;
        }

        const radius = Math.log(dataPoint.ghg_emission);
        const centerX = this.xScale(dataPoint.gdp);
        const centerY = h - this.yScale(dataPoint.meat_consumption);
        const flagScale = 0.7;

        svg.select(`circle[title='${identifier}-${dataPoint.country}']`)
            .transition().duration(250)
            .attr('cx', centerX)
            .attr('cy', centerY)
            .attr('r', radius)
            .attr("fill", color)
            .attr('visibility', 'visible')
            .attr('opacity', this.props.currentHoveredCountry === country ? BubbleChart.OPACITIES.HIGHLIGHTED : BubbleChart.OPACITIES.ENABLED);

        svg.select(`image[title='${identifier}-${dataPoint.country}']`)
            .on("mouseover", () => {
                this.props.hoverCountry(country);
                this.tooltip.show(`<div style="text-align: center"><strong>${dataPoint.country}</strong></div> - ${Math.round(dataPoint.ghg_emission / 1000) + 'K'} greenhouse gas emissions<div> - ${dataPoint.meat_consumption} meat consumed per-capita</div> - ${dataPoint.gdp} GDP per-capita`);
            })
            .on("mousemove", () => this.tooltip.move(d3.event.pageY - 10, d3.event.pageX + 10))
            .on("mouseout", () => {
                this.props.hoverCountry(undefined);
                this.tooltip.hide();
            })
            .on("click", () => this.props.selectCountry(country))
            .transition().duration(250)
            .attr('x', this.props.currentHoveredCountry === country ? centerX - radius : centerX - flagScale * radius)
            .attr('y', this.props.currentHoveredCountry === country ? centerY - radius : centerY - flagScale * radius)
            .attr('height', this.props.currentHoveredCountry === country ? 2 * radius : flagScale * 2 * radius)
            .attr('width', this.props.currentHoveredCountry === country ? 2 * radius : flagScale * 2 * radius)
            .attr('visibility', 'visible')
            .attr('opacity', BubbleChart.OPACITIES.HIGHLIGHTED);

    };
}
