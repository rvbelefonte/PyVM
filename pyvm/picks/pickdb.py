from __future__ import (absolute_import, division, print_function,
        unicode_literals)
from pyvm.db.backends.sqlite3.connection import Connection
from pyvm.db.backends.sqlite3.utils import format_search
from pyvm.picks.schema import TABLES, BASIC_TABLES, BASIC_VIEWS, VMTOMO_VIEWS



class PickDatabase(Connection):

    def __init__(self, database=':memory:', spatial=False, rebuild=False,
            strict_integrity=True):

        Connection.__init__(self, database=database, spatial=spatial)

        # Setup tables
        if not spatial:
            self._init_basic_tables(rebuild=rebuild)
            self.SPATIAL = False
        else:
            #TODO spatialite tables
            msg = "Spatialite support is not yet implemented."
            msg += " Use `spatial=False` for now."
            raise NotImplementedError(msg)

        self._init_tables(rebuild=rebuild)

        # Setup views
        if not self.SPATIAL:
            self._init_basic_views(rebuild=rebuild)
        else:
            #TODO spatialite views
            pass

        self._init_vmtomo_views(rebuild=rebuild)

        # Enforce data integrity
        if strict_integrity:
            self.execute('PRAGMA foreign_keys=ON')

    def _init_tables(self, rebuild=False):
        """
        Initializes tables common to non-Spatialite and Spatialite databases
        """
        for table in TABLES:
            if rebuild:
                self.execute("DROP TABLE IF EXISTS ':table'",
                        {'table': table})
            if table not in self.tables:
                self.execute(TABLES[table])

    def _init_basic_tables(self, rebuild=False):
        """
        Initializes basic pick database tables.

        These tables are used by non-spatialite databases.
        """
        for table in BASIC_TABLES:
            if rebuild:
                self.execute("DROP TABLE IF EXISTS ':table'",
                        {'table': table})
            if table not in self.tables:
                self.execute(BASIC_TABLES[table])

    def _init_basic_views(self, rebuild=False):
        """
        Initializes basic pick database views.

        These views are used by non-spatialite databases.
        """
        for view in BASIC_VIEWS:
            if rebuild:
                self.execute("DROP VIEW IF EXISTS ':view'",
                        {'view': view})
            if view not in self.views:
                self.execute(BASIC_VIEWS[view])

    def _init_vmtomo_views(self, rebuild=False):
        """
        Initialize views for use by VM Tomography
        """
        for view in VMTOMO_VIEWS:
            if rebuild:
                self.execute("DROP VIEW IF EXISTS ':view'",
                        {'view': view})
            if view not in self.views:
                self.execute(VMTOMO_VIEWS[view])

    def _get_events(self):
        return self.read_table('events')
    events = property(fget=_get_events)

    def _get_sources(self):
        return self.read_table('sources')
    sources = property(fget=_get_sources)

    def _get_receivers(self):
        return self.read_table('receivers')
    receivers = property(fget=_get_receivers)

    def _get_picks(self):
        return self.read_table('picks')
    picks = property(fget=_get_picks)

    def add_event(self, event, branchid=0, subid=0, description='',
            replace=False):
        """
        Adds an event to the event table.

        Parameters
        ----------
        event: str
            Event name (e.g., "Pn").
        branchid: int, optional
            Integer branchid ID that specifies what layer the arrival is
            associated with. Default is 0.
        subid: int, optional
            #TODO
        description: str, optional
            Description of the event (e.g. "Reflection from Moho.")
        replace: bool, optional
            Determines whether or not to update the event record if it
            already exists.  Default (False) is to raise an error if the
            event already exists.
        """
        sql = "INSERT"
        if replace:
            sql += " OR REPLACE"
        sql += " INTO 'events' (event, branchid, subid, description)"
        sql += " VALUES (?, ?, ?, ?)"

        self.execute(sql, (event, branchid, subid, description))

    def add_source(self, srcid, srcx, srcy, srcz, replace=False):
        """
        Adds a source to the sources table

        Parameters
        ----------
        srcid: int
            Unique integer ID.
        srcx, srcy: float
            Easting and northing of the source. If Spatialite support is
            enabled, these coordinates must be in the coordinate
            reference system set by PickDatabase.SRID.
        srcz: float
            Depth of the source.
        replace: bool, optional
            Determines whether or not to update the source record if it
            already exists.  Default (False) is to raise an error if the
            source point already exists.
        """
        if not self.SPATIAL:
            sql = "INSERT"
            if replace:
                sql += " OR REPLACE"
            sql += " INTO 'sources' (srcid, srcx, srcy, srcz)"
            sql += " VALUES (?, ?, ?, ?)"
            self.execute(sql, (srcid, srcx, srcy, srcz))
        else:
            raise NotImplementedError

    def add_receiver(self, recid, recx, recy, recz, replace=False):
        """
        Adds a receiver to the receivers table

        Parameters
        ----------
        recid: int
            Unique integer ID.
        recx, recy: float
            Easting and northing of the receiver. If Spatialite support is
            enabled, these coordinates must be in the coordinate
            reference system set by PickDatabase.SRID.
        recz: float
            Depth of the receiver.
        replace: bool, optional
            Determines whether or not to update the receiver record if it
            already exists.  Default (False) is to raise an error if the
            receiver point already exists.
        """
        if not self.SPATIAL:
            sql = "INSERT"
            if replace:
                sql += " OR REPLACE"
            sql += " INTO 'receivers' (recid, recx, recy, recz)"
            sql += " VALUES (?, ?, ?, ?)"
            self.execute(sql, (recid, recx, recy, recz))
        else:
            raise NotImplementedError
    
    def add_pick(self, event, srcid, recid, time, error=0.0,
            replace=False):
        """
        Adds or replaces a pick.

        Parameters
        ----------
        event: str
            Event name (e.g., 'Pn').  Must already exist in the 'events'
            table.
        srcid: int
            Source-point ID.  Must already exist in the 'sources' table.
        recid: int
            Receiver-point ID.  Must already exist in the 'receivers'
            table.
        time: float
            Traveltime of the pick.
        error: float, optional
            Traveltime error for the pick.  Default is 0.0.
        replace: bool, optional
            If `True`, replace existing pick with the same event, srcid,
            recid, if it exists.  If `False`, an IntegrityError is raised
            a pick with the same event, srcid, recid already exists.
        """
        sql = "INSERT"
        if replace:
            sql += " OR REPLACE"
        sql += " INTO 'picks' (event, srcid, recid, time, error)"
        sql += " VALUES (?, ?, ?, ?, ?)"
        self.execute(sql, (event, srcid, recid, time, error))

    def to_vmtomo(self, sources_file=None, receivers_file=None,
            picks_file=None, header=False, sep='\t', **kwargs):
        """
        Formats pick data for input to the underlying tomography code.

        Parameters
        ----------
        sources_file: str or buffer
            Filename or buffer to write the sources data to. If `None`
            (default), no source data is written.
        receivers_file: str or buffer
            Filename or buffer to write the receivers data to. If `None`
            (default), no receiver data is written.
        picks_file: str or buffer
            Filename or buffer to write the pick data to. If `None`
            (default), no pick data is written.
        header: bool, optional
            Determines whether or not to write header rows. A header row
            cannot be present in files read by the underlying tomography
            code. Default is to not write a header.
        sep: str
            Field delimiter for the output file.
        kwargs, optional
            Keyword arguments for selecting picks from the database. Default
            is to include all picks.

        Returns
        -------
        sources, receviers, picks: str
            Strings of formatted source, recevier, and pick data.
        """
        if len(kwargs) > 0:
            search = ' ' + format_search(kwargs)
        else:
            search = ''

        # sources
        sql = "SELECT DISTINCT srcid, srcx, srcy, srcz"
        if search == '':
            sql += " FROM sources"
        else:
            sql += " FROM master_picks"
            sql += " WHERE " + search
        sql += " ORDER BY srcid"
        sources = self.read_sql(sql).to_csv(path_or_buf=sources_file,
                sep=str(sep), index=False, header=header)

        # receivers
        sql = "SELECT DISTINCT recid, recx, recy, recz"
        if search == '':
            sql += " FROM receivers"
        else:
            sql += " FROM master_picks"
            sql += " WHERE " + search
        sql += " ORDER BY recid"
        receivers = self.read_sql(sql).to_csv(path_or_buf=receivers_file,
                sep=str(sep), index=False, header=header)

        # picks
        sql = "SELECT recid, srcid, branchid, subid, offset, time, error"
        sql += " FROM master_picks"
        if search != '':
            sql += " WHERE " + search
        picks = self.read_sql(sql).to_csv(path_or_buf=picks_file,
                sep=str(sep), index=False, header=header)

        return sources, receivers, picks
