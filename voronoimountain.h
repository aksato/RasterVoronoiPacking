#ifndef VORONOIMOUNTAIN_H
#define VORONOIMOUNTAIN_H
#include<QImage>
#include<QPoint>
#include<QHash>
#include <QSharedData>

class VoronoiMountainData : public QSharedData
{
  public:
    VoronoiMountainData() { }
    VoronoiMountainData(const VoronoiMountainData &other)
        : QSharedData(other), origin(other.origin), image(other.image) { }
    ~VoronoiMountainData() { }

    QPoint origin;
    QImage image;
};

class VoronoiMountain
{
public:
    VoronoiMountain() { d = new VoronoiMountainData; }
    VoronoiMountain(QImage image, QPoint origin) {
        d = new VoronoiMountainData;
        setOrigin(origin);
        setImage(image);
    }
    VoronoiMountain(const VoronoiMountain &other)
          : d (other.d)
    {
    }

    void setOrigin(QPoint origin) {d->origin = origin;}
    void setImage(QImage image) {d->image = image;}

    QPoint getOrigin() {return d->origin;}
    int getOriginX() {return d->origin.x();}
    int getOriginY() {return d->origin.y();}
    QImage getImage() {return d->image;}

private:
  QSharedDataPointer<VoronoiMountainData> d;
};

#endif // VORONOIMOUNTAIN_H
