import { ComponentFixture, TestBed } from '@angular/core/testing';

import { SeiteDatabaseComponent } from './seite-database.component';

describe('SeiteDatabaseComponent', () => {
  let component: SeiteDatabaseComponent;
  let fixture: ComponentFixture<SeiteDatabaseComponent>;

  beforeEach(() => {
    TestBed.configureTestingModule({
      declarations: [SeiteDatabaseComponent]
    });
    fixture = TestBed.createComponent(SeiteDatabaseComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
